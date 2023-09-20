# 总述


- 本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的参赛题目，选题为：用TensorRT-LLM实现新模型；为TensorRT-LLM添加新feature，或者在模型上启用了现有feature
    - 原始模型的名称：[Qwen-7b-chat](https://huggingface.co/Qwen/Qwen-7b-chat)
    - Qwen-7b-chat 是一个参数量为7B的模型，使用来自公开的超过2.2万亿tokens的数据进行训练，语言方面主要为中英文，和一般的模型区别如下：1.使用untied embedding嵌入；2.使用RoPE相对位置编码；3.使用RMSNorm替换LayerNorm；4.FFN激活函数SwiGLU代替ReLU；5.采用flash Attention。
- 优化效果  
  - TensorRT-LLM构建的Qwen-7b-chat测试结果如下：  
TensorRT-LLM (total latency: 31.374691486358643 sec)  
TensorRT-LLM beam 0 result  
rouge1 : 15.03897779266612  
rouge2 : 3.6833363780294732  
rougeL : 11.771691444354222  
rougeLsum : 13.216607576575065  
  - 原始Qwen-7b-chat模型测试结果如下：  
  Hugging Face (total latency: 81.77836441993713 sec)  
  HF beam 0 result  
  rouge1 : 15.224433387052777  
  rouge2 : 3.6904330608242164  
  rougeL : 10.85424486688883  
  rougeLsum : 12.84889226768845  
  
  运行测试脚本summarize.py，total latency从81.77836441993713 sec降低到31.374691486358643 sec，加速比约为2.6，优化后的模型rouge score与原始模型的rouge score差距在1以内。
- 运行步骤  
  - 运行环境：  
  使用官方提供的镜像：registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1  
  TensorRT 9.0.0.2  
  CUDA 12.1  
  python 3.8.10  
  NVIDIA driver 525.105.17  
  - 此外需要安装以下python库：  
  tiktoken-0.5.1  
  transformers-stream-generator-0.0.4  
  datasets-2.14.5  
  nltk-3.8.1  
  rouge-score-0.1.2  
  - 文件准备  
  模型文件放置于容器内路径：/root/test/Qwen-7b-Chat  
  代码文件：在/root/test下运行
    ```
    git clone https://github.com/zuocebianpingmao/tensorrt_llm_july.git
    ```
    替换部分python文件：  
    ```
    cp /root/test/tensorrt_llm_july/tensorrt_llm_july-release-v1/tensorrt_llm/models/__init__.py /usr/local/lib/python3.8/dist-packages/tensorrt_llm/models/
    cp -r /root/test/tensorrt_llm_july/tensorrt_llm_july-release-v1/tensorrt_llm/models/qwen /usr/local/lib/python3.8/dist-packages/tensorrt_llm/models/
    ```
  - 运行  
    1.将hugging face模型转换格式：  
    ```
    cd /root/test/tensorrt_llm_july/tensorrt_llm_july-release-v1/examples/qwen
    python hf_qwen_convert.py -i /root/test/Qwen-7b-Chat \
                              -o ./c-model/qwen-7b \
                              --tensor-parallelism 1 \
                              --storage-type float16 \
                              --processes 1
    ```  
    2.构建模型，生成engine文件：
    ```
    python build.py --model_dir ./c-model/qwen-7b/1-gpu \
                    --dtype float16 \
                    --use_gpt_attention_plugin float16 \
                    --use_gemm_plugin float16 \
                    --output_dir trtModel \
                    --enable_context_fmha
    ```  
    3.运行测试：  
    ```
    # 测试tensorrt-llm
    python summarize.py --engine_dir trtModel \
                        --batch_size 1 \
                        --test_trt_llm \
                        --hf_model_location /root/test/Qwen-7b-Chat \
                        --check_accuracy \
                        --tensorrt_llm_rouge1_threshold 14
    # 测试huggingface模型
    python summarize.py --engine_dir trtModel \
                        --batch_size 1 \
                        --test_hf \
                        --hf_model_location /root/test/Qwen-7b-Chat \
                        --check_accuracy \
                        --tensorrt_llm_rouge1_threshold 14
    ```

# 主要开发工作

## 开发工作的难点
- 由于tensorrt-llm无法像tensorrt一样，通过解析onnx自动完成网络构建，所以需要利用tensorrt-llm的API手动搭建网络，这就需要正确的理解原始模型的各个组成部分，任何一个部分未能准确还原都会导致完全错误的结果
- Qwen-7b-Chat与一般模型相比具有以下特点：1.使用untied embedding嵌入；2.使用RoPE相对位置编码；3.使用RMSNorm替换了通常使用的LayerNorm；4.FFN激活函数SwiGLU代替ReLU；5.采用flash Attention。
- 为了方便模型的构建，将模型拆分成一些基础的组成部分，需要实现以下内容：  
  - Attention模块
  - Block模块：模型主体部分由32个Block堆叠组成
  - Model：模型主体，包含32个Block和一些embedding层，norm层
  - CausalLM：包含模型主体和线性层，用于生成输入tensor的prepare_input函数
- 开发过程中不可避免的要对精度进行对齐，需要获取某些中间层的输出。但目前tensorrt-llm的debug的方式较为麻烦，需要在模型代码中使用register_network_output将tensor注册为模型的输出，然后重新构建engine文件，最后在tensorrt_llm_july-release-v1/tensorrt_llm/runtime/generation.py中打印输出。实际使用过程中费时费力。
- 缺乏文档和说明，不知道tensorrt-llm如何使用，很多class或function花费了较长时间才理解了如何使用。

# 开发与优化过程
## 构建模型的主要流程
在敲下第一行代码之前需要明确这一阶段的首要目标，即实现一个可以流畅运行的版本，模型精度可以暂时不做考虑，只要模型可以输出通顺的句子即可，精度可以后续进行优化。此版本的代码位于feature-v1分支。
### 格式转换脚本
首先实现格式转换脚本，代码位于tensorrt_llm_july-release-v1/examples/qwen/目录下的hf_qwen_convert.py和convert.py，主要作用为将Qwen-7b-chat的原始模型转换为FasterTransformer格式以及对某些层的命名作一些调整。Qwen-7b-chat模型相关的代码在tensorrt_llm_july-release-v1/examples/qwen/model_utils
### 模型文件
这一部分使用tensorrt-llm完成模型代码的编写，位置：tensorrt_llm_july-release-v1/tensorrt_llm/models/qwen/model.py。包含了QwenDecoderLayer，QwenModel，QwenForCausalLM 3个类，这里只需要参照models下面的gpt等模型和Qwen-7b-chat原始模型将各种层添加到正确位置，就可以将代码写出来  
需要注意的是： 
- QwenDecoderLayer：模型共有32个QwenDecoderLayer，实现参考了gpt，gptj等模型。Attention层直接使用了tensorrt-llm提供的Attention类。
- Attention的bias设置为True，但Attention内部dense的bias要设置为False，所以这里直接将dense进行替换
  ```
  self.attention.dense = RowLinear(hidden_size,
                                  hidden_size,
                                  bias=False,
                                  dtype=dtype,
                                  tp_group=tp_group,
                                  tp_size=tp_size)
  ```
- 与常见的模型有所区别，normlization使用的是RmsNorm，mlp使用GatedMLP，而且mlp_hidden_size需要除2才是正确的shape
  ```
  self.mlp = GatedMLP(hidden_size=hidden_size,
                      ffn_hidden_size=mlp_hidden_size // 2,
                      hidden_act=hidden_act,
                      dtype=dtype,
                      bias=False,
                      tp_group=tp_group,
                      tp_size=tp_size)
  ```
### 生成engine文件
在转换格式之后需要生成tensorrt运行所需要的engine文件，代码位于tensorrt_llm_july-release-v1/examples/qwen/目录下的build.py和weight.py。运行的时候需要添加参数--use_gpt_attention_plugin float16，--use_gemm_plugin float16，--enable_context_fmha，具体可以参考文档开头的运行步骤
### 运行测试
在生成engine文件之后，可以运行一下模型看看效果。代码在：tensorrt_llm_july-release-v1/examples/qwen/run.py，运行方式：
```
python run.py --tokenizer_dir /root/test/Qwen-7b-Chat \
              --input_text 自然哲学的数学原理 \
              --engine_dir trtModel
```
注意:
- 输入和输出的编解码过程用到了tensorrt_llm_july-release-v1/examples/qwen/model_utils/qwen_generation_utils.py中的 decode_tokens, make_context函数，使用方式可以参考tensorrt_llm_july-release-v1/examples/qwen/model_utils/modeling_qwen.py中的QWenLMHeadModel.chat函数
- SamplingConfig的end_id和pad_id需要设置为151643
```
END_ID=151643
sampling_config = SamplingConfig(end_id=END_ID, pad_id=END_ID, num_beams=1)
```
### 计算rouge score
最后运行一下summarize.py，计算rouge score，total latency以评估模型的精度和性能，代码在tensorrt_llm_july-release-v1/examples/qwen/summarize.py，运行方式参照文档开头的运行步骤，计算结果如下：  
- **原始Qwen-7b-chat模型测试结果：**  
  Hugging Face (total latency: 81.77836441993713 sec)  
  HF beam 0 result  
  rouge1 : 15.224433387052777  
  rouge2 : 3.6904330608242164  
  rougeL : 10.85424486688883  
  rougeLsum : 12.84889226768845 
- **TensorRT-LLM构建的Qwen-7b-chat测试结果：**  
TensorRT-LLM (total latency: 32.793442487716675 sec)  
TensorRT-LLM beam 0 result  
rouge1 : 12.27131682013105  
rouge2 : 4.527097520249083  
rougeL : 9.070103602909926  
rougeLsum : 10.177149178334947
  
加速比约为2.49，rouge score与原始模型尚有一定的差距

## 优化
### 完善模型细节
通过上面的rouge score计算可以看出当前版本存在一定的精度损失。在对比了tensorrt-llm的Attention类和Qwen-7b-chat模型的QWenAttention类就能发现，Attention不能完全复原QwenAttention的计算过程，缺失了一些关键的细节，需要进行改写。改写后的类命名为QwenAttention，改写主要是增加了关于rotary position embedding的计算，除了对Attention类进行改写之外，hf_qwen_convert.py， weight.py也需要进行一些调整；
#### 添加position embedding weight
在hf_qwen_convert.py添加了关于position embedding的计算，并将其保存成.bin文件
```
nMaxSL = 2048
inv_freq = 10**(-1 / 32 * np.arange(0, 128, 2, dtype=np.float32))
valueTable = np.matmul(
    np.arange(nMaxSL, dtype=np.float32).reshape(-1, 1),
    np.concatenate([inv_freq, inv_freq],
                    axis=0).reshape(1, -1)).reshape(nMaxSL,
                                                    len(inv_freq) * 2)
np.cos(valueTable).astype(storage_type).tofile(saved_dir /
                                                "model.cosTable.weight.bin")
np.sin(valueTable).astype(storage_type).tofile(saved_dir /
                                                "model.sinTable.weight.bin")
```
这部分计算可以参考tensorrt_llm_july-release-v1/examples/qwen/model_utils/modeling_qwen.py中的RotaryEmbedding
#### Attention类改写
在QwenModel类的__init__中添加两个embedding层：position_embedding_cos，position_embedding_sin
```
self.position_embedding_cos = Embedding(max_position_embeddings,
                                        self.half_head_size,
                                        dtype=dtype)
self.position_embedding_sin = Embedding(max_position_embeddings,
                                        self.half_head_size,
                                        dtype=dtype)
```
相应的在weight.py中增加从文件中加载这部分权重的代码
```
tensorrt_llm_qwen.position_embedding_cos.weight.value = (fromfile(
        dir_path, 'model.cosTable.weight.bin',
        [n_positions, kv_channel]))
tensorrt_llm_qwen.position_embedding_sin.weight.value = (fromfile(
    dir_path, 'model.sinTable.weight.bin',
    [n_positions, kv_channel]))
```
QwenModel的forward函数中计算position_embedding
```
batch_size = shape(input_ids.data, 0)
input_len = shape(input_ids.data, 1)

hidden_states = self.embedding(input_ids.data)

position_embedding_cos = self.position_embedding_cos(position_ids)
position_embedding_sin = self.position_embedding_sin(position_ids)

position_embedding_cos = position_embedding_cos.view(
    concat([batch_size, input_len, 1, self.half_head_size])
)

position_embedding_sin = position_embedding_sin.view(
    concat([batch_size, input_len, 1, self.half_head_size])
)

position_embedding = [
    position_embedding_cos, position_embedding_sin
]
```
QwenAttention的forward函数中主要增加了rotary position embedding的计算，这部分可以参考tensorrt_llm_july-release-v1/examples/qwen/model_utils/modeling_qwen.py中的apply_rotary_pos_emb和_rotate_half
```
query, key, value = split(qkv, hidden_states.size()[-1], dim=2)
        
query = query.view(
    concat([
        shape(qkv, 0),
        shape(qkv, 1), self.num_attention_heads,
        self.attention_head_size
    ]))
key = key.view(
    concat([
        shape(qkv, 0),
        shape(qkv, 1), self.num_attention_heads,
        self.attention_head_size
    ]))
value = value.view(
    concat([
        shape(qkv, 0),
        shape(qkv, 1), self.num_attention_heads,
        self.attention_head_size
    ]))
zero = constant(
    np.ascontiguousarray(
        np.zeros([1, 1, 1, 1],
                  dtype=np.float16
                  if self.dtype == trt.float16 else np.float32)))

def rotate(x64):
    x32_part0, x32_part1 = x64.split(64, dim=-1)

    x32_part1_negtive = zero - x32_part1

    y64 = concat([x32_part1_negtive, x32_part0], dim=3)
    return y64

def rotate_embedding(x, position_embedding_value):
    cos0, sin0 = position_embedding_value
    x128 = x
    x64_part0, x64_part1 = x128.split((cos0.size()[-1], x128.size()[-1]-cos0.size()[-1]), dim=-1)

    x64_part0_rotate = rotate(x64_part0)
    y64_part0 = x64_part0 * cos0 + x64_part0_rotate * sin0

    y128 = concat([y64_part0, x64_part1], dim=3)
    y128 = y128.view(shape(x))
    return y128

query = rotate_embedding(query, position_embedding)
key = rotate_embedding(key, position_embedding)
```
### int8 kv cache
为了提高模型性能，考虑启用int8 kv cache，主要用到convert.py中的generate_int8，write_int8和smoothquant.py中的capture_activation_range  
需要修改以下内容：
- convert.py中保存attention.query_key_value.weight这部分，val的形状需要修改
```
elif "attention.query_key_value.weight" in key:
hidden_dim = val.shape[0] // 3
local_dim = val.shape[-1]

# val = val.reshape(3, hidden_dim, local_dim)
val = val.reshape(hidden_dim, 3, local_dim)
split_dim = -1
split_vals = np.split(val, factor, axis=split_dim)
save_split(split_vals, saved_dir, key, i, factor)
if save_int8:
    base_key = key.replace(".weight", "")
    vals_i8 = generate_int8(val, act_range, is_qkv=True)
    write_int8(vals_i8, saved_dir, base_key, split_dim, i, factor)
```
- capture_activation_range中给act_scales[name]["w"]赋值的时候dim设置为1
```
def stat_input_hook(m, x, y, name):
    if isinstance(x, tuple):
        x = x[0]
    stat_tensor(name, x, act_scales, "x")
    stat_tensor(name, y, act_scales, "y")

    # if act_scales[name]["w"] is None:
    #     act_scales[name]["w"] = m.weight.abs().clip(1e-8,
    #                                                 None).max(dim=0)[0]
    if act_scales[name]["w"] is None:
        act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                    None).max(dim=1)[0]
```
- 添加--calibrate-kv-cache，重新运行转换格式
```
python hf_qwen_convert.py -i /root/test/Qwen-7b-Chat \
                          -o ./c-model/qwen-7b \
                          --tensor-parallelism 1 \
                          --storage-type float16 \
                          --processes 1 \
                          --calibrate-kv-cache
```
- 添加--int8_kv_cache，重新生成engine
```
python build.py --model_dir ./c-model/qwen-7b/1-gpu \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir trtModel \
                --enable_context_fmha \
                --int8_kv_cache
```

# 优化效果
## 运行环境：  
使用官方提供的镜像：registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1  
TensorRT 9.0.0.2  
CUDA 12.1  
python 3.8.10  
NVIDIA driver 525.105.17  
GPU: A10，显存24GB
## 测试
- **TensorRT-LLM构建的Qwen-7b-chat（未启用int8 kv cache等）：**  
TensorRT-LLM (total latency: 31.374691486358643 sec)  
TensorRT-LLM beam 0 result  
rouge1 : 15.03897779266612  
rouge2 : 3.6833363780294732  
rougeL : 11.771691444354222  
rougeLsum : 13.216607576575065  
- **原始Qwen-7b-chat模型测试结果如下：**  
Hugging Face (total latency: 81.77836441993713 sec)  
HF beam 0 result  
rouge1 : 15.224433387052777  
rouge2 : 3.6904330608242164  
rougeL : 10.85424486688883  
rougeLsum : 12.84889226768845  

- **启用int8 kv cache的测试结果：**  
TensorRT-LLM (total latency: 27.99358320236206 sec)  
TensorRT-LLM beam 0 result  
rouge1 : 14.030726068576666  
rouge2 : 3.244306418219462  
rougeL : 11.027431582093367  
rougeLsum : 11.942382117033334
- 性能  
  - 在完成Attention类的修改之后，total latency从81.77836441993713 sec降低到31.374691486358643 sec，加速比约为2.6，相比初始版本优化模型的加速比2.49，略有提升。
  - 启用int8 kv cache后提升了性能，total latency从未启用时的31.374691486358643 sec降低到27.99358320236206 sec；加速比从2.6提高到2.92
- 精度  
  - 对比rouge score，完成Attention类的修改之后，优化模型与原始模型的差距均在1以内，相比初始版本的优化模型精度也有所提高
  - 启用int8 kv cache后精度有所下降

# 送分题答案

- python3 run.py --max_output_len=8，输出：chef and eventually became a chef at a
- python3 summarize.py --engine_dir trt_engine/gpt2/fp16/1-gpu \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --hf_model_location=gpt2 \
                     --check_accuracy \
                     --tensorrt_llm_rouge1_threshold=14  
  输出：  
  TensorRT-LLM (total latency: 3.0252530574798584 sec)  
  TensorRT-LLM beam 0 result  
  rouge1 : 21.869322054781037  
  rouge2 : 6.258925475911645  
  rougeL : 16.755771650012953  
  rougeLsum : 18.68034777724496  
  Hugging Face (total latency: 14.61320185661316 sec)  
  HF beam 0 result  
  rouge1 : 18.182978950152904  
  rouge2 : 5.166241888544473  
  rougeL : 14.851620358520162  
  ougeLsum : 16.9575774841227
