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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.65it/s]


    2026-05-13 05:54:29,367 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 05:54:29] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:05,  1.18s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:05,  1.18s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:05,  1.18s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:05,  1.18s/it]Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:04<01:05,  1.18s/it]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:20,  2.52it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:20,  2.52it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:20,  2.52it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:20,  2.52it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:04<00:20,  2.52it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:04<00:20,  2.52it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:04<00:20,  2.52it/s]Compiling num tokens (num_tokens=2816):  12%|█▏        | 7/58 [00:04<00:20,  2.52it/s]Compiling num tokens (num_tokens=2560):  12%|█▏        | 7/58 [00:04<00:20,  2.52it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.98it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=832):  26%|██▌       | 15/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=768):  26%|██▌       | 15/58 [00:04<00:06,  6.98it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:02, 13.96it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:02, 13.96it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:02, 13.96it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:02, 13.96it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:02, 13.96it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:02, 13.96it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:02, 13.96it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:04<00:02, 13.96it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 19.37it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 19.37it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 19.37it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 19.37it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:01, 19.37it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:04<00:01, 19.37it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:04<00:01, 19.37it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 19.37it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 19.37it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:05<00:01, 19.37it/s]Compiling num tokens (num_tokens=176):  55%|█████▌    | 32/58 [00:05<00:01, 19.37it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 29.03it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 29.03it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 29.03it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 29.03it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 29.03it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 29.03it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 29.03it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 29.03it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 29.03it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 29.03it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 37.91it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 37.91it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 37.91it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 37.91it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 37.91it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 37.91it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 37.91it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 37.91it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=51.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=51.35 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=7168 avail_mem=51.34 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=51.34 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=51.34 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=51.34 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=51.33 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=5120 avail_mem=51.32 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=51.32 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=51.32 GB):   9%|▊         | 5/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=51.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.28it/s]Capturing num tokens (num_tokens=3840 avail_mem=51.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=51.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=51.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.28it/s]Capturing num tokens (num_tokens=3072 avail_mem=51.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.28it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=51.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=51.31 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=51.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=51.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=51.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=51.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=51.29 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=51.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=51.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=51.27 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s]Capturing num tokens (num_tokens=960 avail_mem=51.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s] Capturing num tokens (num_tokens=896 avail_mem=51.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s]

    Capturing num tokens (num_tokens=832 avail_mem=51.27 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.16it/s]Capturing num tokens (num_tokens=832 avail_mem=51.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=768 avail_mem=51.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=704 avail_mem=51.27 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=640 avail_mem=51.26 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=576 avail_mem=51.26 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=512 avail_mem=51.25 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=512 avail_mem=51.25 GB):  50%|█████     | 29/58 [00:00<00:00, 43.60it/s]Capturing num tokens (num_tokens=480 avail_mem=51.26 GB):  50%|█████     | 29/58 [00:00<00:00, 43.60it/s]Capturing num tokens (num_tokens=448 avail_mem=51.26 GB):  50%|█████     | 29/58 [00:00<00:00, 43.60it/s]Capturing num tokens (num_tokens=416 avail_mem=51.26 GB):  50%|█████     | 29/58 [00:00<00:00, 43.60it/s]Capturing num tokens (num_tokens=384 avail_mem=51.26 GB):  50%|█████     | 29/58 [00:00<00:00, 43.60it/s]

    Capturing num tokens (num_tokens=352 avail_mem=51.25 GB):  50%|█████     | 29/58 [00:00<00:00, 43.60it/s]Capturing num tokens (num_tokens=352 avail_mem=51.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=320 avail_mem=51.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=288 avail_mem=51.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=256 avail_mem=51.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=240 avail_mem=51.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=224 avail_mem=51.23 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=224 avail_mem=51.23 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.33it/s]Capturing num tokens (num_tokens=208 avail_mem=51.23 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.33it/s]Capturing num tokens (num_tokens=192 avail_mem=51.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.33it/s]Capturing num tokens (num_tokens=176 avail_mem=51.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.33it/s]Capturing num tokens (num_tokens=160 avail_mem=51.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.33it/s]

    Capturing num tokens (num_tokens=144 avail_mem=51.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.33it/s]Capturing num tokens (num_tokens=144 avail_mem=51.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.15it/s]Capturing num tokens (num_tokens=128 avail_mem=51.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.15it/s]Capturing num tokens (num_tokens=112 avail_mem=51.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.15it/s]Capturing num tokens (num_tokens=96 avail_mem=51.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.15it/s] Capturing num tokens (num_tokens=80 avail_mem=51.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.15it/s]Capturing num tokens (num_tokens=64 avail_mem=51.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.15it/s]Capturing num tokens (num_tokens=64 avail_mem=51.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.87it/s]Capturing num tokens (num_tokens=48 avail_mem=51.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.87it/s]Capturing num tokens (num_tokens=32 avail_mem=51.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.87it/s]Capturing num tokens (num_tokens=28 avail_mem=51.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.87it/s]Capturing num tokens (num_tokens=24 avail_mem=51.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.87it/s]

    Capturing num tokens (num_tokens=20 avail_mem=51.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.87it/s]Capturing num tokens (num_tokens=20 avail_mem=51.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=16 avail_mem=51.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=12 avail_mem=51.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=8 avail_mem=51.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.12it/s] Capturing num tokens (num_tokens=4 avail_mem=51.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=4 avail_mem=51.18 GB): 100%|██████████| 58/58 [00:01<00:00, 42.00it/s]


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
    Generated text:  Mary and I'm a 16 year old girl. I'm going to graduate from high school tomorrow and I'm not sure how to make friends. I have been thinking of joining a club or something and I'm wondering what would be best for me. 
    
    Since I am the youngest student at my school, I would like to be a part of a club that would be easier for me to join and would not require a lot of work from me. Also, it would not cost anything. It would also be something that I could participate in for a year, which will give me a chance to make friends. I'm also a
    ===============================
    Prompt: The president of the United States is
    Generated text:  32 years older than the president of Brazil. The president of Brazil is 2 times older than the president of France. If the president of the United States is currently 68 years old, how old would the president of Brazil be in 10 years?
    To solve this problem, we need to determine the current ages of the presidents of the United States, Brazil, and France, and then calculate how old Brazil would be in 10 years based on their current ages.
    
    First, let's denote the current age of the president of Brazil as \( B \). According to the problem, the president of Brazil is 
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. London
    C. Moscow
    D. Rome
    Answer: A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Moscow
    D. Rome
    Answer: A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Moscow
    D. Rome
    Answer: A
    
    What is the capital of France? A. Paris B. London C. Moscow D. Rome
    Answer: A
    
    The capital of France is ____. 
    A. Paris 
    B. London 
    C. Moscow 
    D. Rome 
    Answer: A
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain. But the technology itself is not, and the technology that is being developed is actually quite smart. The key to making things work is to keep the code and the data in order. That means making sure that they are clean, well-formatted, and well-architected. This is the most difficult part of the process, and also the hardest part to get right. I’ve seen some folks who’ve written some really great stuff, but the code is a mess. It’s easy to just write code that works, but it’s hard to keep it working.
    The internet has been a lot of our data, but most


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [industry]. I'm a [job title] at [company name], and I'm always looking for ways to [describe a new skill or initiative]. I'm always eager to learn and grow, and I'm always looking for opportunities to contribute to the company's success. What's your favorite hobby or activity? I love [mention a hobby or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. Paris is also a major center for art, music, and literature, and is home to many famous museums, theaters, and restaurants. The city is known for its vibrant nightlife and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city that has played a significant role in the development of French culture and is a UNESCO World
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. More advanced hardware: As AI technology continues to advance, we can expect to see more powerful hardware that can process and analyze large amounts of data more efficiently.
    
    3. Increased use of AI in healthcare: AI is already being used in
    


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
    Generated text:  [Name] and I am a [Field of Work]. [Name] loves to [Job Description or Task]. I enjoy [Why I like this job or task]. As [Name], I am passionate [What makes me so passionate about this job/Task]. I am excited to [What I hope to achieve in this role/Task]. I have worked [Number of Years] years in [Field of Work]. [Name] has a [Value] reputation in [Field of Work]. I am confident [Why I believe in this role/Task]. I am looking forward to [Why I am excited to work with you]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Its name comes from the Latin word for "light," and it has a rich history dating back to ancient times. Paris is known for its stunning architecture, vibrant culture, and annual festivals that celebrate its many cultural and historical landmarks. Despite its size, Paris is home to a diverse population and is a bustling metropolis. Its status as the capital also includes several important international organizations and landmarks. In addition to being the cultural and political capital of France, Paris is also a major tourist destination. The city is known for its fashion, gastronomy, and dance. It is often called the "City of Light" and is home to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve many different trends that could shape the direction and impact of this technology in the coming years. Here are some of the key trends that could be expected in the next decade:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and it is likely to continue to be used in the future to improve patient care. Predictive analytics and machine learning algorithms could be used to help identify risk factors for diseases, predict patient outcomes, and develop personalized treatment plans.
    
    2. Increased Use of AI in Manufacturing: AI is already being used in manufacturing to automate processes, improve quality control


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

     Emily

    .

     I

    'm

     a

     self

    -employed

     freelance

     writer

     and

     journalist

    .

     I

    'm

     passionate

     about

     writing

     about

     diverse

     perspectives

     and

     trying

     to

     connect

     with

     people

     through

     my

     work

    .

     My

     goal

     is

     to

     share

     stories

     and

     experiences

     with

     readers

    .

     
    


    I

    'm

     a

     hard

     worker

     and

     always

     strive

     to

     improve

     my

     skills

    .

     I

     enjoy

     meeting

     new

     people

     and

     learning

     from

     them

    .

     I

    'm

     committed

     to

     being

     a

     role

     model

     for

     other

     writers

     and

     journalists

     to

     follow

    .

     
    


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

     evolve

     as

     a

     writer

     and

     journalist

    .

     I

    'm

     excited

     to

     continue

     learning

     and

     exploring

     my

     passion

     for

     writing

    .

     
    


    Thank

     you

     for

     taking

     the

     time

     to

     meet

     me

    .

     I

    'm

     looking

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     

    1

    9

    th

     largest

     city

     in

     the

     world

    .

     It

     was

     founded

     in

     

    8

    6

    9

     AD

     and

     is

     located

     in

     the

     Î

    le

    -de

    -F

    rance

     region

     on

     the

     northern

     bank

     of

     the

     Se

    ine

     River

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     culinary

     traditions

    .

     It

     has

     a

     population

     of

     over

     

    2

    .

     

    3

     million

     people

     and

     is

     the

     third

     most

     populous

     city

     in

     the

     European

     Union

     after

     London

     and

     Rome

    .

     Paris

     is

     also

     one

     of

     the

     world

    ’s

     top

     tourist

     destinations

    ,

     known

     for

     its

     museums

    ,

     theaters

    ,

     and

     fashion

    .

     The

     city

     is

     home

     to

     numerous

     cultural

     institutions

     and

     landmarks

    ,

     such

     as

     the

     Lou

    vre

     Museum

    ,

     Notre

    -D

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     complex

     and

     rapidly

     evolving

     landscape

     with

     a

     variety

     of

     potential

     trends

     and

     advancements

     that

     could

     shape

     the

     way

     we

     live

    ,

     work

    ,

     and

     interact

     with

     technology

    .

     Some

     of

     the

     key

     trends

     and

     technologies

     that

     are

     likely

     to

     be

     significant

     contributors

     to

     AI

     in

     the

     coming

     years

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     There

     is

     growing

     concern

     about

     the

     impact

     of

     AI

     on

     society

     and

     the

     environment

    ,

     and

     the

     need

     for

     companies

     and

     governments

     to

     take

     steps

     to

     ensure

     that

     AI

     systems

     are

     developed

     and

     used

     in

     a

     responsible

     and

     ethical

     manner

    .
    


    2

    .

     Adv

    ancements

     in

     machine

     learning

     and

     deep

     learning

    :

     AI

     systems

     are

     becoming

     more

     sophisticated

     and

     able

     to

     learn

     from

     vast

     amounts

     of

    



```python
llm.shutdown()
```
