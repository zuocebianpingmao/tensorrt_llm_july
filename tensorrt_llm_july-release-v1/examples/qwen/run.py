import argparse
import json
import os

import tensorrt_llm
import torch
import transformers
from model_utils.qwen_generation_utils import decode_tokens, make_context
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from transformers import GenerationConfig

from build import get_engine_name  # isort:skip

END_ID = 151643

_ERROR_BAD_CHAT_FORMAT = """\
We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format in generation_config is not "chatml".
If you are directly using the model downloaded from Huggingface, please make sure you are using our "Qwen/Qwen-7B-Chat" Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
我们检测到您可能在使用预训练模型（而非chat模型）进行多轮chat，因为您当前在generation_config指定的chat_format，并未设置为我们在对话中所支持的"chatml"格式。
如果您在直接使用我们从Huggingface提供的模型，请确保您在调用model.chat()时，使用的是"Qwen/Qwen-7B-Chat"模型（而非"Qwen/Qwen-7B"预训练模型）。
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='trtModel')
    parser.add_argument(
        '--input_text',
        type=str,
        default=
        'Continuation: Nvidia was founded on April 5, 1993, by Jensen Huang，')
    parser.add_argument(
        '--input_tokens',
        type=str,
        help='CSV file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default='/root/Qwen-7b-Chat',
                        help='Directory containing the tokenizer model.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)

    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    multi_query_mode = config['builder_config']['multi_query_mode']

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('qwen', dtype, world_size, runtime_rank)
    serialize_path = os.path.join(args.engine_dir, engine_name)
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_dir, trust_remote_code=True)
    
    generation_config = GenerationConfig.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT
    history = []

    max_window_size = generation_config.max_window_size
    raw_text, context_tokens = make_context(
        tokenizer,
        args.input_text,
        history=history,
        system="You are a helpful assistant.",
        max_window_size=max_window_size,
        chat_format=generation_config.chat_format,
    )


    input_ids = torch.tensor([context_tokens]).int().contiguous().cuda()
    input_lengths = torch.tensor(
        [input_ids.size(1) for _ in range(input_ids.size(0))]).int().cuda()
    
    model_config = ModelConfig(model_name="qwen",
                               num_heads=num_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               multi_query_mode=multi_query_mode,
                               remove_input_padding=remove_input_padding
                               )

    sampling_config = SamplingConfig(end_id=END_ID, pad_id=END_ID, num_beams=1)
    decoder = tensorrt_llm.runtime.GenerationSession(
        model_config, engine_buffer, runtime_mapping, debug_mode=True)
    decoder.setup(input_ids.size(0), input_ids.size(1), args.max_output_len)
    outputs = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()


    response = decode_tokens(
            outputs[0][0],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=generation_config.chat_format,
            verbose=False,
            errors='replace'
        )
    print(f'Input --->\n {args.input_text}')
    print(f'Output --->\n {response}')

    print("Finished!")
