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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.39it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.38it/s]


    2026-05-08 01:42:54,814 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 01:42:54] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:23,  5.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:23,  5.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:23,  5.68s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:23,  5.68s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:23,  5.68s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:46,  1.15it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:46,  1.15it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:46,  1.15it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:46,  1.15it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:46,  1.15it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:46,  1.15it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:46,  1.15it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:15,  3.13it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:15,  3.13it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:15,  3.13it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:15,  3.13it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:15,  3.13it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:15,  3.13it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:15,  3.13it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:15,  3.13it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:15,  3.13it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:06<00:05,  6.64it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:06<00:05,  6.64it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:06<00:05,  6.64it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:06<00:05,  6.64it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:06<00:05,  6.64it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:06<00:05,  6.64it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:06<00:05,  6.64it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:06<00:05,  6.64it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:06<00:03, 10.51it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:06<00:03, 10.51it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:06<00:03, 10.51it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:06<00:03, 10.51it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:06<00:03, 10.51it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:06<00:03, 10.51it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:06<00:03, 10.51it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:06<00:03, 10.51it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:06<00:03, 10.51it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:06<00:01, 16.00it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:06<00:01, 16.00it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:06<00:01, 16.00it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:06<00:01, 16.00it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:06<00:01, 16.00it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:06<00:01, 16.00it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:06<00:01, 16.00it/s]

    Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:06<00:01, 16.00it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:06<00:01, 16.00it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:06<00:00, 22.39it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:06<00:00, 22.39it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:06<00:00, 22.39it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:06<00:00, 22.39it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:06<00:00, 22.39it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:06<00:00, 22.39it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:06<00:00, 22.39it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:06<00:00, 22.39it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:06<00:00, 22.39it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:06<00:00, 29.10it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:06<00:00, 29.10it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:06<00:00, 29.10it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:06<00:00, 29.10it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:06<00:00, 29.10it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:06<00:00, 29.10it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:06<00:00, 29.10it/s]

    Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:06<00:00, 29.10it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:06<00:00, 29.10it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 16.69it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.17it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.17it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.99it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.32it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.32it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.32it/s]

    Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.32it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.32it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.32it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  50%|█████     | 29/58 [00:00<00:00, 36.85it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 36.85it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 36.85it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 36.85it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.68it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.17it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.17it/s] Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 38.40it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 38.40it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 38.40it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 38.40it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 38.40it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.17it/s]Capturing num tokens (num_tokens=28 avail_mem=74.34 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.17it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.17it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.17it/s]Capturing num tokens (num_tokens=16 avail_mem=74.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.17it/s]Capturing num tokens (num_tokens=16 avail_mem=74.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 32.50it/s]Capturing num tokens (num_tokens=12 avail_mem=74.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 32.50it/s]Capturing num tokens (num_tokens=8 avail_mem=74.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 32.50it/s] Capturing num tokens (num_tokens=4 avail_mem=74.23 GB):  95%|█████████▍| 55/58 [00:01<00:00, 32.50it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.23 GB): 100%|██████████| 58/58 [00:01<00:00, 32.94it/s]


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
    Generated text:  Alvin and I am a 15 year old aspiring artist. I live in a small town in the middle of the western part of the United States. I am a 4th grader and I love drawing and painting. I draw in watercolors and pastels and I paint in acrylics. I am a fan of all art and I try my best to improve in all areas of my art.
    I am also a passionate animal lover and I love to take care of the animals and trying to keep them healthy. I have a pet dog named pipsack and i like to take care of him. I am also really
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. A. 错误 B. 正确
    错误
    
    关于朗肯循环，下列说法错误的是______。 A. 能将蒸汽的热能转变为电能 B. 能将蒸汽的热能转变为水的热能 C. 可以将水的热能转化为机械能 D. 能将蒸汽的热能转变为液体的热能
    能将蒸汽的热能转变为电能
    
    某市的居民是按照自己的家庭状况分成了三类：家庭收入低于1万元的居民、家庭收入在1万元至5万元的居民和家庭收入高于
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris, the former capital of France and the birthplace of French literature and the arts.
    The city's most famous building, the Louvre, is one of the world's most important museums. The city also has the Notre-Dame Cathedral, the iconic pink iron bell tower, and the Arc de Triomphe. Another historic landmark is the Eiffel Tower, completed in 1889. Paris, as a city of philosophy, is famous for its literary, cultural, and artistic atmosphere. Paris is one of the largest and most populous cities in the world, with a population of over 6 million.
    
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly changing as it evolves and is constantly evolving. With the rapid adoption of AI, the use of AI in a variety of fields is increasing. AI has the potential to revolutionize the world by improving efficiency, reducing costs, and creating new opportunities. However, it also poses a number of ethical and social concerns. Some concerns include bias in AI, privacy issues, and the potential impact on the job market. These concerns are significant and require a comprehensive approach to addressing the challenges of AI development. The integration of AI into society requires a thoughtful and nuanced approach to balancing the benefits and risks of the technology. This paper explores the ethical and social


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and music, and is home to many world-renowned museums, theaters, and art galleries. Paris is a bustling city with a vibrant culture and a rich history, making it a popular tourist destination. 
    
    Paris is the capital of France and is known for its iconic landmarks such as the Eiffel Tower
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and robotics: As AI continues to advance, we are likely to see an increase in automation and robotics in various industries. This will lead to the development of new jobs and the displacement of existing ones, but it will also create new opportunities for people to work in areas such as data analysis, machine learning, and robotics.
    
    2. Enhanced privacy and security: As AI becomes more advanced, we are likely
    


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
    Generated text:  [Name], and I'm an experienced [field or occupation]. I'm confident in my ability to communicate effectively with people and am passionate about [what you're passionate about]. I enjoy working with people to find solutions and help them achieve their goals. I have a deep understanding of [related topic or area], and I'm committed to learning and growing every day. I'm eager to assist you and help you reach your full potential.
    What is your profession, and what do you enjoy most about it? As an experienced [field or occupation], I'm passionate about [what you're passionate about]. I enjoy working with people to find solutions
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The statement is concise, directly referring to the capital city of France. It is 3 words long and provides a clear and specific answer to the question. The statement can be easily understood and conveyed. It does not contain any grammatical errors or typos. The answer is accurate and relevant to the question asked. 
    Question: What is the capital of France? 
    
    Answer: Paris. 
    The answer is accurate and relevant to the question asked. It is concise, directly referring to the capital city of France. It is 3 words long and provides a clear and specific answer to the question. The statement can be easily
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and constantly evolving, with new advancements and emerging technologies driving innovation and change. Some possible future trends in artificial intelligence include:
    
    1. Self-learning and self-improving: One of the key areas of AI research is developing algorithms that can learn and adapt from data, rather than just following pre-programmed instructions. This could lead to more efficient and effective AI systems that can handle complex and changing situations.
    
    2. Increased use of AI in healthcare: AI is already being used to help doctors diagnose and treat diseases more accurately and quickly than human physicians. As AI improves, it's likely to become even more prevalent in healthcare, potentially revolutionizing


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

     __

    __.

     I

    'm

     a

    /an

     __

    __.

     I

    've

     been

     working

    /

    learning

     __

    __.

     I

     enjoy

     __

    __.

     I

    'm

     an

     ___

     who

     is

     passionate

     about

     __

    __.

     I

    'm

     excited

     to

     start

     this

     journey

    ,

     and

     I

    'm

     looking

     forward

     to

     meeting

     new

     people

     and

     learning

     something

     new

    .

     Thanks

     for

     asking

    !

     Let

    's

     make

     this

     a

     good

     start

     to

     our

     friendship

    .

     Who

    's

     the

     other

     person

     in

     this

     sentence

    ?

     Write

     your

     answer

     in

     the

     provided

     template

    .

     Person

     Name

    :

     __________________________________

    ________________

    ____

    ___

    


    Short

     and

     Neutral

     Self

    -

    Introduction

    :

     __________________________________

    ________________

    ____

    ___

    


    Neutral

     Topic

    :

     __________________________________

    ________________

    ____

    ___

    


    Why

     is

     this

     topic

     relevant

    :

     __________________________________

    ________________

    ____

    ___

    


    Connection

     to

     character

    :

     __________________________________

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     historical

     significance

    ,

     rich

     culture

    ,

     and

     stunning

     architecture

    .
    


    What

     is

     the

     capital

     city

     of

     France

    ?

     Paris

    .

     It

     is

     known

     for

     its

     historical

     significance

    ,

     rich

     culture

    ,

     and

     stunning

     architecture

    .

     
    


    D

    etermine

     whether

     the

     following

     statement

     is

     true

     or

     false

    :

     "

    The

     capital

     city

     of

     France

     is

     named

     after

     the

     first

     king

     of

     France

    ."

     No

    ,

     the

     capital

     city

     of

     France

     is

     not

     named

     after

     the

     first

     king

     of

     France

    .

     The

     capital

     city

     of

     France

     is

     actually

     called

     "

    Paris

    ,"

     which

     is

     derived

     from

     the

     Latin

     "

    par

    vis

    ,"

     meaning

     "

    alley

    ,"

     and

     "

    dom

    us

    ,"

     meaning

     "

    house

    ."

     The

     city

     was

     founded

     by

     the

     Romans

     and

     is

     now

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     rapidly

     changing

    ,

     with

     many

     possibilities

     emerging

     as

     technology

     advances

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     everyday

     life

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     may

     see

     more

     widespread

     use

     of

     AI

     in

     areas

     such

     as

     healthcare

    ,

     education

    ,

     and

     transportation

    .

     This

     could

     lead

     to

     even

     more

     personalized

     and

     efficient

     solutions

     to

     real

    -world

     problems

    .
    


    2

    .

     Greater

     focus

     on

     ethical

     AI

    :

     As

     AI

     becomes

     more

     prevalent

     in

     our

     lives

    ,

     there

     will

     be

     increased

     pressure

     to

     ensure

     that

     AI

     is

     developed

     and

     used

     in

     a

     way

     that

     align

    s

     with

     ethical

     standards

    .

     This

     could

     mean

     more

     focus

     on

    



```python
llm.shutdown()
```
