'''
Convert huggingface Qwen model. Use https://huggingface.co/Qwen/Qwen-7B-chat as demo.
'''
import argparse
import configparser
import dataclasses
import os
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
from smoothquant import capture_activation_range, smooth_gemm
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from model_utils.modeling_qwen import QWenBlock
from convert import split_and_save_weight

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy


@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 4
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "qwen"
    storage_type: str = "fp32"
    dataset_cache_dir: str = None

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--out-dir',
                            '-o',
                            type=str,
                            help='file name of output directory',
                            required=True)
        parser.add_argument('--in-file',
                            '-i',
                            type=str,
                            help='file name of input checkpoint file',
                            required=True)
        parser.add_argument('--tensor-parallelism',
                            '-tp',
                            type=int,
                            help='Requested tensor parallelism for inference',
                            default=1)
        parser.add_argument(
            "--processes",
            "-p",
            type=int,
            help=
            "How many processes to spawn for conversion (default: 4). Set it to a lower value to reduce RAM usage.",
            default=4)
        parser.add_argument(
            "--calibrate-kv-cache",
            "-kv",
            action="store_true",
            help=
            "Generate scaling factors for KV cache. Used for storing KV cache in int8."
        )
        parser.add_argument(
            "--smoothquant",
            "-sq",
            type=float,
            default=None,
            help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
            " to Smoothquant the model, and output int8 weights."
            " A good first try is 0.5. Must be in [0, 1]")
        parser.add_argument(
            "--model",
            default="qwen",
            type=str,
            help="Specify GPT variants to convert checkpoints correctly",
            # choices=["gpt2", "santacoder", "starcoder"]
            )
        parser.add_argument("--storage-type",
                            "-t",
                            type=str,
                            default="float32",
                            choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--dataset-cache-dir",
                            type=str,
                            default=None,
                            help="cache dir to load the hugging face dataset")
        return ProgArgs(**vars(parser.parse_args(args)))


@torch.no_grad()
def smooth_qwen_model(model, scales, alpha):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, QWenBlock):
            continue

        # qkv_proj
        layer_name = name + ".attn.c_attn"
        smoother = smooth_gemm(module.attn.c_attn.weight.T,
                               scales[layer_name]["x"], module.ln_1.weight,
                               module.ln_1.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(dim=0)[0]

        # fc1
        layer_name = name + ".mlp.c_fc"
        smoother = smooth_gemm(module.mlp.c_fc.weight.T,
                               scales[layer_name]["x"], module.ln_2.weight,
                               module.ln_2.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.c_fc.weight.abs().max(dim=0)[0]


def qwen_to_ft_name(orig_name):
    global_weights = {
        "transformer.wte.weight": "model.wte",
        "transformer.ln_f.weight": "model.final_layernorm.weight",
        "lm_head.weight": "model.lm_head.weight"
    }

    if orig_name in global_weights:
        return global_weights[orig_name]
    
    _, _, layer_id, *weight_name = orig_name.split(".")
    layer_id = int(layer_id)
    weight_name = "transformer." + ".".join(weight_name)

    per_layer_weights = {
        "transformer.ln_1.weight": "input_layernorm.weight",
        "transformer.attn.c_attn.bias": "attention.query_key_value.bias",
        "transformer.attn.c_attn.weight": "attention.query_key_value.weight",
        "transformer.attn.c_proj.weight": "attention.dense.weight",
        "transformer.ln_2.weight": "post_attention_layernorm.weight",
        "transformer.mlp.w1.weight": "mlp.dense_h_to_4h.gate.weight",
        "transformer.mlp.w2.weight": "mlp.dense_h_to_4h.weight",
        "transformer.mlp.c_proj.weight": "mlp.dense_4h_to_h.weight",
    }

    return f"layers.{layer_id}.{per_layer_weights[weight_name]}"


@torch.no_grad()
def hf_qwen_converter(args: ProgArgs):
    infer_tp = args.tensor_parallelism
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # load position_embedding from rank 0
    model = AutoModelForCausalLM.from_pretrained(args.in_file,
                                                 device_map="auto",
                                                 trust_remote_code=True,
                                                 fp16=True)
    act_range = {}
    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        from datasets import load_dataset
        dataset = load_dataset("lambada",
                               split="validation",
                               cache_dir=args.dataset_cache_dir)
        act_range = capture_activation_range(
            model, AutoTokenizer.from_pretrained(args.in_file), dataset)
        if args.smoothquant is not None:
            smooth_qwen_model(model, act_range, args.smoothquant)

    config = configparser.ConfigParser()
    config["qwen"] = {}
    for key in vars(args):
        config["qwen"][key] = f"{vars(args)[key]}"
    for k, v in vars(model.config).items():
        config["qwen"][k] = f"{v}"
    config["qwen"]["storage_dtype"] = args.storage_type

    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_dtype_to_torch(args.storage_type)

    global_ft_weights = [
        "model.wte", "model.final_layernorm.weight", 
        "model.lm_head.weight"
    ]


    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    starmap_args = []
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            continue
        ft_name = qwen_to_ft_name(name)

        if ft_name in global_ft_weights:
            torch_to_numpy(param.to(storage_type).cpu()).tofile(
                saved_dir / f"{ft_name}.bin")
        else:
            param = torch_to_numpy(param.to(storage_type).cpu())
            starmap_args.append(
                (0, saved_dir, infer_tp, ft_name, 
                 param,
                 act_range.get(name.replace(".weight", ""))
                 ))

    starmap_args = tqdm(starmap_args, desc="saving weights")
    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)
    else:
        # simpler for debug situations
        for starmap_arg in starmap_args:
            split_and_save_weight(*starmap_arg)


def run_conversion(args: ProgArgs):
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")
    hf_qwen_converter(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run_conversion(ProgArgs.parse())
    # python3 hf_qwen_convert.py -i /root/Qwen-7b-Chat -o ./c-model/qwen-7b --tensor-parallelism 1 --storage-type float16
