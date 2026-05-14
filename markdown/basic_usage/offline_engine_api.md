# Offline Engine API

SGLang provides a direct inference engine without the need for an HTTP server, especially for use cases where additional HTTP server adds unnecessary complexity or overhead. Here are two general use cases:

- Offline Batch Inference
- Custom Server on Top of the Engine

This document focuses on the offline batch inference, demonstrating four different inference modes:

- Non-streaming synchronous generation
- Streaming synchronous generation
- Non-streaming asynchronous generation
- Streaming asynchronous generation

Additionally, you can easily build a custom server on top of the SGLang offline engine. A detailed example working in a python script can be found in [custom_server](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/custom_server.py).



## Nest Asyncio
Note that if you want to use **Offline Engine** in ipython or some other nested loop code, you need to add the following code:
```python
import nest_asyncio

nest_asyncio.apply()

```

## Advanced Usage

The engine supports [vlm inference](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference_vlm.py) as well as [extracting hidden states](https://github.com/sgl-project/sglang/blob/main/examples/runtime/hidden_states). 

Please see [the examples](https://github.com/sgl-project/sglang/tree/main/examples/runtime/engine) for further use cases.

## Offline Batch Inference

SGLang offline engine supports batch inference with efficient scheduling.


```python
# launch the offline engine
import asyncio

import sglang as sgl
import sglang.test.doc_patch  # noqa: F401
from sglang.utils import async_stream_and_merge, stream_and_merge

llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
```

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.49it/s]


    2026-05-14 00:59:34,446 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 00:59:34] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:14,  3.39it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:14,  3.39it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:14,  3.39it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:14,  3.39it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:14,  3.39it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:14,  3.39it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:14,  3.39it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:04<00:14,  3.39it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:04<00:14,  3.39it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:04<00:14,  3.39it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.15it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:04,  8.15it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 14.03it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 14.03it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 14.03it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 14.03it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 14.03it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 14.03it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 14.03it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 14.03it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 14.03it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:05<00:02, 14.03it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 21.04it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 21.04it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 21.04it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 21.04it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 21.04it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 21.04it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 21.04it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 21.04it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 21.04it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:00, 21.04it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:05<00:00, 21.04it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.91it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.91it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 39.78it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 39.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=49.52 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.48 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.48 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=49.48 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=6656 avail_mem=49.48 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=49.48 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=49.48 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=5632 avail_mem=49.47 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=5120 avail_mem=49.46 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=4608 avail_mem=49.46 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=4608 avail_mem=49.46 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=49.46 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.98it/s]Capturing num tokens (num_tokens=3840 avail_mem=49.46 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.98it/s]Capturing num tokens (num_tokens=3584 avail_mem=49.45 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=49.45 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.98it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=49.45 GB):  21%|██        | 12/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=49.44 GB):  21%|██        | 12/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=49.44 GB):  21%|██        | 12/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=49.44 GB):  21%|██        | 12/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=49.44 GB):  21%|██        | 12/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=49.43 GB):  21%|██        | 12/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=49.43 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s]Capturing num tokens (num_tokens=1792 avail_mem=49.43 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s]Capturing num tokens (num_tokens=1536 avail_mem=49.43 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s]Capturing num tokens (num_tokens=1280 avail_mem=49.42 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=49.40 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s]

    Capturing num tokens (num_tokens=960 avail_mem=49.42 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s] Capturing num tokens (num_tokens=960 avail_mem=49.42 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=896 avail_mem=49.42 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=832 avail_mem=49.41 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=768 avail_mem=49.41 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=704 avail_mem=49.41 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=640 avail_mem=49.40 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=640 avail_mem=49.40 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=576 avail_mem=49.40 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=512 avail_mem=49.39 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=480 avail_mem=49.40 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]

    Capturing num tokens (num_tokens=448 avail_mem=48.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=416 avail_mem=48.58 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=416 avail_mem=48.58 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=384 avail_mem=48.58 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=352 avail_mem=48.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=320 avail_mem=48.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=288 avail_mem=48.56 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=256 avail_mem=48.56 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.95it/s]Capturing num tokens (num_tokens=256 avail_mem=48.56 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=240 avail_mem=48.56 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=224 avail_mem=48.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]

    Capturing num tokens (num_tokens=208 avail_mem=48.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=192 avail_mem=48.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=176 avail_mem=48.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=176 avail_mem=48.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=160 avail_mem=48.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=144 avail_mem=48.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=128 avail_mem=48.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=112 avail_mem=48.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.47it/s]Capturing num tokens (num_tokens=96 avail_mem=48.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.47it/s] Capturing num tokens (num_tokens=96 avail_mem=48.53 GB):  81%|████████  | 47/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=80 avail_mem=48.53 GB):  81%|████████  | 47/58 [00:01<00:00, 42.01it/s]

    Capturing num tokens (num_tokens=64 avail_mem=48.52 GB):  81%|████████  | 47/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=48 avail_mem=48.52 GB):  81%|████████  | 47/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=32 avail_mem=48.52 GB):  81%|████████  | 47/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=28 avail_mem=48.51 GB):  81%|████████  | 47/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=28 avail_mem=48.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=24 avail_mem=48.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=20 avail_mem=48.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=16 avail_mem=48.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=12 avail_mem=48.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=8 avail_mem=48.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.09it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=48.50 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=4 avail_mem=48.49 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=4 avail_mem=48.49 GB): 100%|██████████| 58/58 [00:01<00:00, 38.34it/s]


### Non-streaming Synchronous Generation


```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

    ===============================
    Prompt: Hello, my name is
    Generated text:  Xiao Fei. I'm from Nanjing. I'm a 14-year-old girl. I like watching  cartoons on the Internet. My name is Zhenbo, and I'm from Shanghai. I like to play computer games. I'm twelve years old. I like playing soccer. I have a big family. I have a father, mother, two brothers and a sister. I love my family very much. I have a brother who is four years old. My brother is not in my family. He is in the USA. He is not my brother. He is my cousin. My cousin is 20 years
    ===============================
    Prompt: The president of the United States is
    Generated text:  a popular candidate in a political campaign. The campaign has raised $2 million to get the candidate to campaign. The campaign estimates that the cost of running the candidate will be $3.5 million. If the president of the United States raises another $3 million, the cost of running the candidate will decrease to $3.2 million. If he raises another $1 million, the cost of running the candidate will decrease to $2.7 million. The cost of running the candidate will be $2.5 million if he raises another $0.5 million. How many additional months will it take for the campaign to reach $2
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 正确
    B. 错误
    答案:
    A
    
    企业应制定相应的业务运作程序和操作规程,并确保所有员工接受相应的培训,以识别、评估和控制风险,预防和减少经营过程中所发生的重大事件。
    A. 正确
    B. 错误
    答案:
    A
    
    夏季,户外作业人员应戴好____,防止中暑。
    A. 防毒面具
    B. 防护眼镜
    C. 防护手套
    D. 防护面罩
    答案:
    B
    
    城南车辆段信号系统
    ===============================
    Prompt: The future of AI is
    Generated text:  not linear or smooth, but rather a series of converging paths. It may begin with a technological breakthrough, but the reality of these technologies will be influenced by the broader social, political, and cultural context. Moreover, the path of AI will be very much dependent on the way that developers, researchers, and policymakers take care of their respective roles. As a result, AI may be used for good or for ill, depending on the decisions that the stakeholders make. Ultimately, the future of AI will be determined by the collective actions of the entire community.
    The discussion of AI has a broad scope, and the relationship between different disciplines of computer


### Streaming Synchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {
    "temperature": 0.2,
    "top_p": 0.9,
}

print("\n=== Testing synchronous streaming generation with overlap removal ===\n")

for prompt in prompts:
    print(f"Prompt: {prompt}")
    merged_output = stream_and_merge(llm, prompt, sampling_params)
    print("Generated text:", merged_output)
    print()
```

    
    === Testing synchronous streaming generation with overlap removal ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that serves as the political, cultural, and economic center of the country. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its rich history, including the French Revolution and the Opéra Garnier, and its cuisine, including its famous dishes like croissants and escargot. The city is also known for its fashion industry, with many famous fashion designers and boutiques. Paris is a vibrant and diverse city with a rich cultural heritage that continues to attract visitors from around the world. It is a city that
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, including in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud detection, and trading algorithms.
    


### Non-streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous batch generation ===")


async def main():
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")


asyncio.run(main())
```

    
    === Testing asynchronous batch generation ===


    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text:  ____. I'm a/an ____. I've been at ____. I've been working at ____. I've been the ____. I'm a/an ____. I'm a/an ____. I'm a/an ____. I'm a/an ____. 
    
    Please, feel free to make the introductions as detailed or concise as you like, but try to keep them consistent with the fictional character's persona and job role. What do you think?
    
    Sincerely, [Your Name] [Your Title] [Your Job Position] [Your Role] [Your Role] [Your Role] [Your Role] [Your Role] [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the Hauts-de-Seine region of the central French region. It is the largest city in France and its population is approximately 10 million. The city is known for its historic architecture, vibrant culture, and annual festival celebrations. Paris is a world-renowned culinary and fashion capital and is a major tourist destination. The city is also known for its high standards of living, and is home to some of the world's most famous museums, theaters, and parks. The city is home to a diverse population with French, Belgian, and other European languages spoken. Paris is a major economic center in Europe, and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by an increase in the integration of AI into various industries and applications, as well as a rise in the development of AI-powered technologies that are more complex and sophisticated than those currently available. Some potential trends in AI that are currently expected to play an important role in the future include:
    
    1. Improved AI ethics and accountability: The development of AI systems that are more transparent and accountable for their actions will become more prevalent. This will help to reduce ethical concerns and ensure that AI is used responsibly.
    
    2. Increased AI diversity and inclusiveness: As AI becomes more integrated into various industries, there will be a greater need to ensure


### Streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous streaming generation (no repeats) ===")


async def main():
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated text: ", end="", flush=True)

        # Replace direct calls to async_generate with our custom overlap-aware version
        async for cleaned_chunk in async_stream_and_merge(llm, prompt, sampling_params):
            print(cleaned_chunk, end="", flush=True)

        print()  # New line after each prompt


asyncio.run(main())
```

    
    === Testing asynchronous streaming generation (no repeats) ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text: 

     [

    Name

    ],

     and

     I

    'm

     a

    /an

     [

    Age

    ]

     year

     old

    .

     I

    'm

     from

     [

    Location

    ],

     and

     I

     have

     a

    /an

     [

    Job

     Title

    ]

     at

     [

    Company

    ],

     where

     I

     work

     hard

     and

     strive

     to

     [

    personal

     trait

    ].

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     develop

     my

     skills

    .

     Thank

     you

     for

     asking

    ,

     and

     I

     look

     forward

     to

     meeting

     you

     and

     learning

     more

     about

     you

    .

     [

    Name

    ]

     [

    Work

     Email

    /

    Phone

     Number

    ]

     [

    Company

     Website

    /

    LinkedIn

     Profile

    ]
    


    ---
    


    **

    Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

    'm

     a

    /an

     [

    Age

    ]

     year

     old

    .

     I

    'm

     from

     [

    Location

    ],

     and

     I

     have

     a

    /an

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

    ,

     the

     historical

     center

     of

     France

    ,

     is

     known

     for

     its

     romantic

     architecture

    ,

     vibrant

     culture

    ,

     and

     annual

     celebrations

    ,

     including

     the

     E

    iff

    el

     Tower

    's

     unveiling

    .

     Additionally

    ,

     it

    's

     a

     hub

     for

     art

    ,

     fashion

    ,

     and

     science

    ,

     and

     is

     home

     to

     many

     influential

     museums

    ,

     including

     the

     Lou

    vre

    .

     Paris

     is

     renowned

     for

     its

     cuisine

     and

     fashion

    ,

     and

     is

     a

     global

     destination

     for

     tourists

     and

     exp

    ats

     alike

    .

     The

     city

     is

     also

     home

     to

     the

     French

     Parliament

    ,

     E

    iff

    el

     Tower

    ,

     and

     various

     parks

     and

     gardens

    .

     Paris

     has

     been

     a

     cultural

     and

     political

     center

     for

     over

     

    2

    ,

    0

    0

    0

     years

    .

     It

    's

     a

     city

     with

     a

     rich

     history

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

    ,

     with

     potential

     developments

     in

     areas

     such

     as

    :
    


    1

    .

     Increasing

    ly

     personalized

     and

     context

    -aware

     AI

    :

     As

     AI

     systems

     become

     more

     integrated

     with

     our

     lives

    ,

     we

     may

     see

     an

     increase

     in

     personalized

     and

     context

    -aware

     AI

     that

     learns

     from

     our

     interactions

     with

     the

     system

     and

     adap

    ts

     accordingly

    .
    


    2

    .

     Enhanced

     machine

     learning

     capabilities

    :

     Adv

    ancements

     in

     machine

     learning

     techniques

     may

     lead

     to

     even

     more

     sophisticated

     AI

     systems

     that

     can

     learn

     from

     their

     environment

     and

     make

     more

     accurate

     predictions

     and

     decisions

    .
    


    3

    .

     Increased

     reliance

     on

     AI

     in

     various

     industries

    :

     With

     the

     growing

     global

     emphasis

     on

     automation

     and

     efficiency

    ,

     AI

     systems

     may

     become

     more

     prevalent

     in

     various

     industries

    ,

     including

     healthcare

    ,

     transportation

    ,

     and

     retail

    .
    


    4

    



```python
llm.shutdown()
```
