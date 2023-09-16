import configparser
import time
from pathlib import Path

import numpy as np
import tensorrt_llm
import torch
from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.models import QwenForCausalLM
from tensorrt_llm.quantization import QuantMode


def gen_suffix(rank, use_smooth_quant, quant_per_channel):
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix
    return suffix


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])
    return None


def parse_ft_config(ini_file):
    qwen_config = configparser.ConfigParser()
    qwen_config.read(ini_file)

    n_embd = qwen_config.getint('qwen', 'hidden_size')
    n_head = qwen_config.getint('qwen', 'num_attention_heads')
    n_layer = qwen_config.getint('qwen', 'num_hidden_layers')
    n_positions = qwen_config.getint('qwen', 'seq_length')
    kv_channel = qwen_config.getint('qwen', 'kv_channels')
    vocab_size = qwen_config.getint('qwen', 'vocab_size')
    do_layer_norm_before = qwen_config.getboolean('qwen', 'do_layer_norm_before', fallback=True)
    rotary_pct = qwen_config.getfloat('qwen', 'rotary_pct', fallback=0.0)
    hidden_act = 'silu'
    bias = qwen_config.getboolean('qwen', 'bias', fallback=False)
    inter_size = qwen_config.getint('qwen', 'intermediate_size', fallback=None)

    if inter_size is None:
        inter_size = 4 * n_embd

    multi_query_mode = qwen_config.getboolean('qwen', 'multi_query_mode', fallback=False)
    return n_embd, n_head, n_layer, n_positions, kv_channel, vocab_size, do_layer_norm_before, hidden_act, rotary_pct, bias, inter_size, multi_query_mode


def load_from_ft(tensorrt_llm_qwen: QwenForCausalLM,
                 dir_path,
                 rank=0,
                 tensor_parallel=1,
                 fp16="float16",
                 multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from FT...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_qwen, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    n_embd, n_head, n_layer, n_positions, kv_channel, vocab_size, do_layer_norm_before, hidden_act, rotary_pct, bias, inter_size, multi_query_mode = parse_ft_config(
        Path(dir_path) / 'config.ini')
    np_dtype = np.float16 if fp16 == 'float16' else np.float32

    def fromfile(dir_path, name, shape=None, dtype=None):
        dtype = np_dtype if dtype is None else dtype
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    def set_smoothquant_scale_factors(module,
                                      pre_scale_weight,
                                      dir_path,
                                      basename,
                                      shape,
                                      per_tok_dyn,
                                      per_channel,
                                      is_qkv=False,
                                      rank=None):
        suffix = "bin"
        if per_channel:
            if rank is not None:
                suffix = f"{rank}." + suffix
            suffix = "col." + suffix

        col_shape = shape if (per_channel or is_qkv) else [1, 1]
        if per_tok_dyn:
            if pre_scale_weight is not None:
                pre_scale_weight.value = np.array([1.0], dtype=np.float32)
            t = fromfile(dir_path, f"{basename}scale_w_quant_orig.{suffix}",
                         col_shape, np.float32)
            module.per_channel_scale.value = t
        else:
            t = fromfile(dir_path, f"{basename}scale_x_orig_quant.bin", [1],
                         np.float32)
            pre_scale_weight.value = t
            t = fromfile(dir_path, f"{basename}scale_y_accum_quant.{suffix}",
                         col_shape, np.float32)
            module.per_channel_scale.value = t
            t = fromfile(dir_path, f"{basename}scale_y_quant_orig.bin", [1, 1],
                         np.float32)
            module.act_scale.value = t

    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_qwen, "quant_mode", QuantMode(0))
    # Do we use SmoothQuant?
    use_smooth_quant = quant_mode.has_act_and_weight_quant()
    # Do we use quantization per token?
    quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
    # Do we use quantization per channel?
    quant_per_channel = quant_mode.has_per_channel_scaling()

    # Do we use INT4/INT8 weight-only?
    use_weight_only = quant_mode.is_weight_only()

    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    def sq_trick(x):
        return x.view(np.float32) if use_smooth_quant else x

    # Debug
    suffix = gen_suffix(rank, use_smooth_quant, quant_per_channel)
    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    tensorrt_llm_qwen.position_embedding_cos.weight.value = (fromfile(
        dir_path, 'model.cosTable.weight.bin',
        [n_positions, kv_channel]))
    tensorrt_llm_qwen.position_embedding_sin.weight.value = (fromfile(
        dir_path, 'model.sinTable.weight.bin',
        [n_positions, kv_channel]))

    tensorrt_llm_qwen.embedding.weight.value = (fromfile(
        dir_path, 'model.wte.bin', [vocab_size, n_embd]))
    if do_layer_norm_before:
        tensorrt_llm_qwen.ln_f.weight.value = (fromfile(
            dir_path, 'model.final_layernorm.weight.bin'))
    # share input embedding
    lm_head_weight = fromfile(dir_path, 'model.lm_head.weight.bin',
                              [vocab_size, n_embd])
    if lm_head_weight is None:
        lm_head_weight = fromfile(dir_path, 'model.wte.bin',
                                  [vocab_size, n_embd])
    if vocab_size % tensor_parallel != 0:
        # padding
        vocab_size_padded = tensorrt_llm_qwen.lm_head.out_features * tensor_parallel
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                'constant',
                                constant_values=0)
    tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
        split(lm_head_weight, tensor_parallel, rank))
    for i in range(n_layer):
        tensorrt_llm_qwen.layers[i].input_layernorm.weight.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.weight.' + suffix,
            [3 * n_embd // tensor_parallel, n_embd], w_type)
        if t is not None:
            dst = tensorrt_llm_qwen.layers[i].attention.qkv.weight
            if use_smooth_quant:
                dst.value = sq_trick(
                    np.ascontiguousarray(np.transpose(t, [1, 0])))
                set_smoothquant_scale_factors(
                    tensorrt_llm_qwen.layers[i].attention.qkv,
                    tensorrt_llm_qwen.layers[i].input_layernorm.scale_to_int,
                    dir_path,
                    'model.layers.' + str(i) + '.attention.query_key_value.',
                    [1, 3 * n_embd // tensor_parallel],
                    quant_per_token_dyn,
                    quant_per_channel,
                    rank=rank,
                    is_qkv=True)
            elif use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(t), plugin_weight_only_quant_type)
                # workaround for trt not supporting int8 inputs in plugins currently
                dst.value = processed_torch_weights.view(
                    dtype=torch.float32).numpy()
                scales = tensorrt_llm_qwen.layers[
                    i].attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = np.ascontiguousarray(t)
        
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.bias.' + str(rank) + '.bin')
        if t is not None:
            dst = tensorrt_llm_qwen.layers[i].attention.qkv.bias
            dst.value = np.ascontiguousarray(t)

        dst = tensorrt_llm_qwen.layers[i].attention.dense.weight
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
            [n_embd, n_embd // tensor_parallel], w_type)
        if use_smooth_quant:
            dst.value = sq_trick(np.ascontiguousarray(np.transpose(t, [1, 0])))
            dense_scale = getattr(tensorrt_llm_qwen.layers[i].attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].attention.dense, dense_scale, dir_path,
                'model.layers.' + str(i) + '.attention.dense.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
        elif use_weight_only:
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.view(
                dtype=torch.float32).numpy()
            scales = tensorrt_llm_qwen.layers[i].attention.dense.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            dst.value = np.ascontiguousarray(t)

        if bias:
            dst = tensorrt_llm_qwen.layers[i].attention.dense.bias
            dst.value = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.attention.dense.bias.bin')

        dst = tensorrt_llm_qwen.layers[i].post_layernorm.weight
        dst.value = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.post_attention_layernorm.weight.bin')

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.dense_h_to_4h.weight.' + suffix,
            [inter_size // 2 // tensor_parallel, n_embd], w_type)
        if use_smooth_quant:
            tensorrt_llm_qwen.layers[i].mlp.fc.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].mlp.fc,
                tensorrt_llm_qwen.layers[i].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.dense_h_to_4h.',
                [1, inter_size // 2 // tensor_parallel],
                quant_per_token_dyn,
                quant_per_channel,
                rank=rank)
        elif use_weight_only:
            dst = tensorrt_llm_qwen.layers[i].mlp.fc.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.view(
                dtype=torch.float32).numpy()
            scales = tensorrt_llm_qwen.layers[i].mlp.fc.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_qwen.layers[i].mlp.fc.weight.value = np.ascontiguousarray(
                t)
        if bias:
            tensorrt_llm_qwen.layers[i].mlp.fc.bias.value = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.mlp.dense_h_to_4h.bias.' + str(rank) + '.bin')
        if non_gated_version(hidden_act):
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.mlp.dense_h_to_4h.gate.weight.' + suffix,
                [inter_size // 2 // tensor_parallel, n_embd], w_type)
            tensorrt_llm_qwen.layers[
                i].mlp.gate.weight.value = np.ascontiguousarray(t)

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.dense_4h_to_h.weight.' + suffix,
            [n_embd, inter_size // 2 // tensor_parallel], w_type)
        if use_smooth_quant:
            tensorrt_llm_qwen.layers[i].mlp.proj.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            proj_scale = getattr(tensorrt_llm_qwen.layers[i].mlp,
                                 "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_qwen.layers[i].mlp.proj, proj_scale, dir_path,
                'model.layers.' + str(i) + '.mlp.dense_4h_to_h.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
        elif use_weight_only:
            dst = tensorrt_llm_qwen.layers[i].mlp.proj.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.view(
                dtype=torch.float32).numpy()
            scales = tensorrt_llm_qwen.layers[i].mlp.proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_qwen.layers[
                i].mlp.proj.weight.value = np.ascontiguousarray(t)
        if bias:
            tensorrt_llm_qwen.layers[i].mlp.proj.bias.value = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.mlp.dense_4h_to_h.bias.bin')

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            tensorrt_llm_qwen.layers[
                i].attention.kv_orig_quant_scale.value = 1.0 / t
            tensorrt_llm_qwen.layers[i].attention.kv_quant_orig_scale.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def load_from_hf_qwen(tensorrt_llm_qwen: QwenForCausalLM,
                           hf_qwen,
                           rank=0,
                           tensor_parallel=1,
                           fp16=False):
    tensorrt_llm.logger.info('Loading weights from HF Qwen...')
    tik = time.time()

    valid_lm_head_weight = False
    for k, v in hf_qwen.state_dict().items():
        torch_dtype = torch.float16 if fp16 == 'float16' else torch.float32
        v = v.to(torch_dtype).cpu().numpy()
        if 'wte.weight' in k:
            tensorrt_llm_qwen.embedding.weight.value = v
        elif 'wpe.weight' in k:
            tensorrt_llm_qwen.embedding.position_embedding.weight.value = v
        elif 'ln_f.weight' in k:
            tensorrt_llm_qwen.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
                split(v, tensor_parallel, rank))
            valid_lm_head_weight = True
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if 'ln_1.weight' in k:
                tensorrt_llm_qwen.layers[idx].input_layernorm.weight.value = v

            elif 'attn.c_attn.weight' in k:
                v = v.transpose()
                dst = tensorrt_llm_qwen.layers[idx].attention.qkv.weight
                dst.value = np.ascontiguousarray(split(v, tensor_parallel,
                                                       rank))
            elif 'attn.c_attn.bias' in k:
                dst = tensorrt_llm_qwen.layers[idx].attention.qkv.bias
                dst.value = np.ascontiguousarray(split(v, tensor_parallel,
                                                       rank))
            elif 'attn.c_proj.weight' in k:
                v = v.transpose()
                dst = tensorrt_llm_qwen.layers[idx].attention.dense.weight
                dst.value = np.ascontiguousarray(
                    split(v, tensor_parallel, rank, dim=1))
                dst.value = v
            elif 'ln_2.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].post_layernorm.weight
                dst.value = v
            elif 'mlp.c_fc.weight' in k:
                v = v.transpose()
                tensorrt_llm_qwen.layers[
                    idx].mlp.fc.weight.value = np.ascontiguousarray(
                        split(v, tensor_parallel, rank))
            elif 'mlp.c_proj.weight' in k:
                v = v.transpose()
                tensorrt_llm_qwen.layers[
                    idx].mlp.proj.weight.value = np.ascontiguousarray(
                        split(v, tensor_parallel, rank, dim=1))

    if not valid_lm_head_weight:
        # Use wte as lm_head weight to match the load_from_ft implementation.
        lm_head_weight = tensorrt_llm_qwen.embedding.vocab_embedding.weight._value
        vocab_size = hf_qwen.config.vocab_size
        if vocab_size % tensor_parallel != 0:
            # padding
            vocab_size_padded = tensorrt_llm_qwen.lm_head.out_features * tensor_parallel
            pad_width = vocab_size_padded - vocab_size
            lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                    'constant',
                                    constant_values=0)
        tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, tensor_parallel, rank))

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
