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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.56it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.55it/s]


    2026-05-10 17:08:01,935 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 17:08:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.15it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.15it/s]

    Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 11.83it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 11.83it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 11.83it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 11.83it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 11.83it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 11.83it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 11.83it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 16.13it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 16.13it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 16.13it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 16.13it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 16.13it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 16.13it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 16.13it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 16.13it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 22.26it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 25.60it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 25.60it/s]

    Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 25.60it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 25.60it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 25.60it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 25.60it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 25.60it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 28.42it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 32.11it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 32.11it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 32.11it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.98 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.94 GB):   3%|▎         | 2/58 [00:00<00:04, 13.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.94 GB):   3%|▎         | 2/58 [00:00<00:04, 13.60it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.92 GB):   3%|▎         | 2/58 [00:00<00:04, 13.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.92 GB):   7%|▋         | 4/58 [00:00<00:03, 15.26it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.93 GB):   7%|▋         | 4/58 [00:00<00:03, 15.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.93 GB):   7%|▋         | 4/58 [00:00<00:03, 15.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.91 GB):   7%|▋         | 4/58 [00:00<00:03, 15.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.91 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.90 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.45it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=71.90 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.89 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.89 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.89 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.88 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.87 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.87 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.87 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.58it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.86 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.58it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=72.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.80 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.80 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.46it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.81 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.80 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.46it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=71.78 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.46it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.78 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.10it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.57 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.10it/s]Capturing num tokens (num_tokens=960 avail_mem=71.76 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.10it/s] Capturing num tokens (num_tokens=896 avail_mem=71.76 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.10it/s]Capturing num tokens (num_tokens=896 avail_mem=71.76 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.87it/s]Capturing num tokens (num_tokens=832 avail_mem=71.75 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.87it/s]

    Capturing num tokens (num_tokens=768 avail_mem=71.75 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.87it/s]Capturing num tokens (num_tokens=704 avail_mem=71.74 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.87it/s]Capturing num tokens (num_tokens=704 avail_mem=71.74 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.89it/s]Capturing num tokens (num_tokens=640 avail_mem=71.72 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.89it/s]Capturing num tokens (num_tokens=576 avail_mem=71.72 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.89it/s]Capturing num tokens (num_tokens=512 avail_mem=71.70 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.89it/s]Capturing num tokens (num_tokens=512 avail_mem=71.70 GB):  50%|█████     | 29/58 [00:01<00:01, 25.15it/s]Capturing num tokens (num_tokens=480 avail_mem=71.71 GB):  50%|█████     | 29/58 [00:01<00:01, 25.15it/s]

    Capturing num tokens (num_tokens=448 avail_mem=71.70 GB):  50%|█████     | 29/58 [00:01<00:01, 25.15it/s]Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:01<00:01, 25.15it/s]Capturing num tokens (num_tokens=384 avail_mem=71.68 GB):  50%|█████     | 29/58 [00:01<00:01, 25.15it/s]Capturing num tokens (num_tokens=384 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.96it/s]Capturing num tokens (num_tokens=352 avail_mem=71.67 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.96it/s]Capturing num tokens (num_tokens=320 avail_mem=71.66 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.96it/s]Capturing num tokens (num_tokens=288 avail_mem=71.65 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.96it/s]Capturing num tokens (num_tokens=256 avail_mem=71.63 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.96it/s]

    Capturing num tokens (num_tokens=256 avail_mem=71.63 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.85it/s]Capturing num tokens (num_tokens=240 avail_mem=71.64 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.85it/s]Capturing num tokens (num_tokens=224 avail_mem=71.62 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.85it/s]Capturing num tokens (num_tokens=208 avail_mem=71.61 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.85it/s]Capturing num tokens (num_tokens=192 avail_mem=71.61 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.85it/s]Capturing num tokens (num_tokens=192 avail_mem=71.61 GB):  71%|███████   | 41/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=176 avail_mem=71.61 GB):  71%|███████   | 41/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=160 avail_mem=71.59 GB):  71%|███████   | 41/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=144 avail_mem=71.59 GB):  71%|███████   | 41/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=128 avail_mem=71.59 GB):  71%|███████   | 41/58 [00:01<00:00, 32.09it/s]

    Capturing num tokens (num_tokens=112 avail_mem=71.59 GB):  71%|███████   | 41/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=112 avail_mem=71.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=96 avail_mem=71.58 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.80it/s] Capturing num tokens (num_tokens=80 avail_mem=71.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=64 avail_mem=71.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=48 avail_mem=71.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=32 avail_mem=71.55 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=32 avail_mem=71.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.16it/s]Capturing num tokens (num_tokens=28 avail_mem=71.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.16it/s]Capturing num tokens (num_tokens=24 avail_mem=71.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.16it/s]Capturing num tokens (num_tokens=20 avail_mem=71.53 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.16it/s]

    Capturing num tokens (num_tokens=16 avail_mem=71.53 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.16it/s]Capturing num tokens (num_tokens=12 avail_mem=71.52 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.16it/s]Capturing num tokens (num_tokens=12 avail_mem=71.52 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=8 avail_mem=71.52 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.58it/s] Capturing num tokens (num_tokens=4 avail_mem=71.51 GB):  97%|█████████▋| 56/58 [00:02<00:00, 38.58it/s]Capturing num tokens (num_tokens=4 avail_mem=71.51 GB): 100%|██████████| 58/58 [00:02<00:00, 28.64it/s]


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
    Generated text:  Cesar and I am a System Software Engineer. What programming language do you use to develop software? As a System Software Engineer, my programming language of choice is Python. Python is widely used in software engineering due to its simplicity, readability, and extensive ecosystem of libraries, frameworks, and tools that facilitate development. Python is particularly useful for tasks like data analysis, web development, artificial intelligence, and more. Additionally, it's widely used for machine learning, automation, and many other tasks. Would you like me to help you with any programming questions or tasks related to software engineering? How can I assist you? Let's get started! �
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military jets he should have on the horizon. The president would like to have 3.0 times as many jets as there are days in a year. The cost to purchase each jet is $x per jet. If the president has a budget of $20 million, how many jets would he be able to purchase?
    To determine how many military jets the president can purchase within his budget, we need to follow these steps:
    
    1. Calculate the number of days in a year.
    2. Determine the number of jets the president would have if he has 3.0 times the number of days in a year
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city in France and the most populous city in Europe. The population is about 2.1 million. It is the capital of France and the most populous city in Europe. The Paris metropolitan area is the second-largest metropolitan area in Europe by population.
    
    The reason why Paris is considered the capital of France is:
    
    A) Because it is the largest city in France
    
    B) Because it is the most populous city in Europe
    
    C) Because it is the largest city in Europe
    
    D) Because it is the capital of France
    
    To determine why Paris is considered the capital of France, we need to consider the
    ===============================
    Prompt: The future of AI is
    Generated text:  predictably uncertain, especially with a looming economic downturn. What exactly does it mean to be a world-leading AI company? Do we need a new metric to measure the importance of AI? This paper explores the current state of AI and how it is changing the way we do business. It will help companies understand the importance of AI in their operations and how to harness AI to make a meaningful impact on society. The first part will outline the current state of AI and how it is changing the way we do business. The second part will explore how AI is impacting various industries and how it is changing business models. The third part will discuss the importance


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a vibrant cultural scene. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its delicious cuisine, including French cuisine, and its annual festivals and events. It is a popular tourist destination and a cultural hub for the country. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. Its status as the capital of France is a testament to its importance and influence on the country. The city is also home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems.
    
    2. Enhanced ethical considerations: As AI becomes more advanced, there will be increasing pressure to address ethical concerns, such as bias, privacy, and transparency. This could lead to more stringent regulations and standards for AI development and deployment.
    
    3. Greater reliance on AI for decision-making: AI is likely to become more integrated into decision-making processes, allowing machines to make more informed and accurate decisions
    


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
    Generated text:  _____. My journey with the Beast of Light began on a whim. In a moment of pure creativity and exploration, I found myself lost in a vast, untamed wilderness, my heart pounding with excitement. I had no idea what I was going to do, but something about the experience was irresistible. With a flick of my wrist, I unleashed a surge of dark magic, and darkness began to engulf the land. I felt my first callous, malevolent heart begin to hum. How could I have known that this would be my destiny? I had no idea what I would be doing in the future, but there was a spark of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is located in the heart of the region and serves as the cultural, economic, and political capital of the country. Paris is a major city with a rich history and has been the capital of France since 1804. It is home to many iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Champs-Élysées. Paris is known for its diverse culture, rich history, and annual festivals, making it a popular tourist destination worldwide. The city has a population of around 2.3 million and is home to a variety of international companies, including Google
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bound to be a rapidly evolving landscape. Here are some potential trends in the AI space:
    
    1. Improved Natural Language Processing: With advancements in AI, we are likely to see increased capabilities for natural language processing, allowing machines to understand and respond to human speech. This could lead to the development of more advanced virtual assistants and chatbots.
    
    2. Enhanced Machine Learning: With more data and training, machine learning algorithms will become even more powerful. This could lead to the development of systems that can learn and adapt to new situations, making them more efficient and effective at their tasks.
    
    3. AI for Healthcare: AI could be used to improve healthcare


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

    Your

     Name

    ],

     and

     I

     specialize

     in

     [

    Your

     specialty

    ].

     Let

    's

     get

     started

    !
    


    You

     are

     the

     founder

     of

     [

    Your

     company

    ].

     We

    're

     experts

     in

     [

    Your

     specialty

    ],

     specializing

     in

     [

    Your

     specialization

    ].

     We

     bring

     a

     unique

     perspective

     that

     combines

     the

     latest

     technologies

     with

     our

     understanding

     of

     the

     industry

    ,

     making

     us

     the

     go

    -to

     experts

     for

     [

    Your

     specialty

    ].

     We

    're

     here

     to

     help

     anyone

     who

     needs

     advice

     or

     support

     in

     [

    Your

     specialty

    ].

     Contact

     us

     today

     to

     learn

     more

    !

     [

    Your

     Name

    ]

     

    🌍

    ✨

    
    


    **

    Note

    :**

     If

     you

     prefer

     to

     be

     a

     bit

     more

     formal

     or

     structured

    ,

     you

     can

     adjust

     the

     introduction

     to

     reflect

     that

    .

     If

     you

     want

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    [

    The

     statement

     should

     be

     

    3

    0

     words

     or

     less

    ,

     and

     should

     be

     an

     accurate

     and

     brief

     description

     of

     the

     capital

     city

     of

     France

    .]

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     culture

    .

     
    


    [

    The

     statement

     should

     be

     gramm

    atically

     correct

    ,

     concise

    ,

     and

     include

     all

     necessary

     details

     to

     accurately

     describe

     Paris

     as

     the

     capital

     of

     France

    .

     Paris

    ,

     known

     as

     "

    la

     Ville

     Fl

    uv

    iale

    ,"

     is

     also

     known

     as

     "

    l

    '

    Î

    le

     de

     France

    "

     and

     is

     a

     UNESCO

     World

     Heritage

     site

    .]

     
    


    The

     capital

     of

     France

     is

     Paris

    .

     [

    3

    0

     words

     or

     less

    ]

     
    


    [

    Note

    :

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     subject

     to

     rapid

     change

     due

     to

     advances

     in

     technology

    ,

     political

     influences

    ,

     and

     economic

     factors

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     in

     healthcare

    :

     AI

     is

     increasingly

     being

     used

     in

     healthcare

     to

     diagnose

     diseases

    ,

     predict

     patient

     outcomes

    ,

     and

     personalize

     treatments

    .

     However

    ,

     the

     use

     of

     AI

     in

     healthcare

     raises

     questions

     about

     data

     privacy

    ,

     bias

    ,

     and

     the

     ethical

     use

     of

     AI

     in

     medical

     procedures

    .
    


    2

    .

     Adv

    ancements

     in

     natural

     language

     processing

    :

     Natural

     language

     processing

     is

     expected

     to

     continue

     to

     improve

    ,

     making

     AI

     more

     capable

     of

     understanding

     and

     generating

     human

    -like

     language

    .

     This

     could

     lead

     to

     more

     personalized

     and

     effective

     customer

     service

     and

     support

    .
    


    3

    



```python
llm.shutdown()
```
