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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.88it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.88it/s]


    2026-05-13 07:43:05,520 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 07:43:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.41it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.86it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.86it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.86it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.86it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.86it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.86it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.86it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.86it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.86it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.86it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.96it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.96it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.96it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.96it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 23.96it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 23.96it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 23.96it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 23.96it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.96it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.96it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.96it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.97it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.97it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.97it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.97it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.97it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.97it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.97it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.19it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.00 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.37it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.37it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.47it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.47it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.47it/s]Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.47it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.47it/s]

    Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.47it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=768 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=448 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=416 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]

    Capturing num tokens (num_tokens=352 avail_mem=70.93 GB):  50%|█████     | 29/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=352 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=256 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=208 avail_mem=70.91 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.64it/s]Capturing num tokens (num_tokens=208 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.25it/s]Capturing num tokens (num_tokens=192 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.25it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.25it/s]

    Capturing num tokens (num_tokens=144 avail_mem=70.90 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=128 avail_mem=70.90 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.25it/s]Capturing num tokens (num_tokens=128 avail_mem=70.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.53it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.53it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.53it/s] Capturing num tokens (num_tokens=80 avail_mem=70.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.53it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.53it/s]Capturing num tokens (num_tokens=48 avail_mem=70.88 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.53it/s]Capturing num tokens (num_tokens=48 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=32 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=24 avail_mem=70.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.51it/s]

    Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.67it/s]Capturing num tokens (num_tokens=12 avail_mem=70.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.67it/s]Capturing num tokens (num_tokens=8 avail_mem=70.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.67it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.67it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 42.38it/s]


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
    Generated text:  Erica, a software developer by profession. I have been programming since childhood and have been working on this project for about a year. I'm a bit overwhelmed right now, can you help me out with my code? I'm having some trouble with my function that calculates the factorial of a number. Can you help me fix the code?
    ```cpp
    #include <iostream>
    #include <cassert>
    #include <cmath>
    
    int factorial(int n) {
        if (n == 0) {
            return 1;
        }
        return n * factorial(n - 1);
    }
    
    int main() {
        int num = 5;
        assert
    ===============================
    Prompt: The president of the United States is
    Generated text:  32 years older than the president of Brazil. The president of Brazil is 2 times older than the president of France. If the president of the United States is currently 32 years old, what is the total of the current ages of the presidents of France and Brazil? To determine the total of the current ages of the presidents of France and Brazil, we need to first identify the ages of each president based on the given information.
    
    1. Identify the age of the president of the United States:
       \[
       \text{Age of the president of the United States} = 32 \text{ years}
      
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Moscow
    D. New York
    Answer:
    
    A
    
    The spirit of the times is a continuous and significant reflection of the times. For instance, the spirit of innovation refers to:
    A. A spirit of inventiveness and uniqueness
    B. A spirit of seeking new ways to make life better
    C. A spirit of continuous striving for progress
    D. A spirit of striving for self-improvement and development
    Answer:
    
    A
    
    Which of the following options is the correct pronunciation of the character 'qǐng'?
    A. qiǎng
    B. qǐng
    
    ===============================
    Prompt: The future of AI is
    Generated text:  ripe for disruption – and its impact on law enforcement is no exception. Law enforcement agencies increasingly turn to artificial intelligence (AI) to help collect, analyze, and share data, which has many potential benefits. For example, AI-powered analytics can help police gather and process more crime data quickly and efficiently, which can help identify patterns that are hidden in plain sight.
    AI has also been used to create augmented reality (AR) technologies that can be used to train law enforcement officers on how to use a police drone, which could have numerous benefits, including the ability to create safer and more effective training programs. However, there are also concerns about the


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


    Generated text:  Paris. It is the largest city in the country and the seat of government for the country. It is also known as the "City of Light" and is famous for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a cultural and economic hub of the country and is home to many world-renowned museums, theaters, and art galleries. It is also a popular tourist destination and a major center for business and finance. Paris is known for its rich history, diverse culture, and vibrant nightlife. It is a city that is constantly evolving and changing, with new developments and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration of AI into everyday life: As AI becomes more integrated into our daily lives, we may see more widespread adoption of AI technologies in areas such as healthcare, transportation, and customer service. This could lead to more efficient and effective use of resources, as well as improved quality of life for individuals and communities.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more advanced, there will be a greater emphasis on ensuring
    


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
    Generated text:  [Name], and I am a [fill in the blank] [age] year old [occupation] with [fill in the blank] education and experience. What is your profession and what are your notable accomplishments?
    
    [Name]: My name is [Name] and I am a [fill in the blank] [age] year old [occupation] with [fill in the blank] education and experience. What is your profession and what are your notable accomplishments? I am an experienced [fill in the blank] [occupation] with [fill in the blank] years of experience in this field. I have [number] years of experience in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the center of the country, serving as the political, economic, and cultural center. It is also known as the City of Love, as it is home to many famous historical sites, including the Eiffel Tower and Notre Dame Cathedral. The city is known for its vibrant art scene, world-renowned cuisine, and rich history, making it a beloved destination for both locals and tourists alike.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and there are several key trends that are likely to shape its development:
    
    1. Deep Learning: With the development of deep learning algorithms, AI will be able to perform tasks that are currently only possible with human intelligence, such as recognizing images and speech. This will make it possible for AI to learn and adapt to new situations and improve its performance over time.
    
    2. Natural Language Processing: AI will be able to understand human language and interpret it in a way that is similar to how humans understand it. This will enable AI to perform tasks such as language translation, question answering, and even writing.
    
    3. Robotics: AI will become


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

     Sarah

    ,

     and

     I

    'm

     a

     software

     engineer

     with

     a

     passion

     for

     innovation

     and

     problem

    -solving

    .

     I

     have

     a

     unique

     perspective

     on

     technology

     that

     I

     bring

     to

     every

     project

     I

     work

     on

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     the

     lives

     of

     those

     I

     work

     with

    .

     I

    'm

     also

     a

     big

     believer

     in

     the

     power

     of

     collaboration

     and

     I

     strive

     to

     work

     with

     people

     who

     are

     open

     to

     new

     ideas

     and

     perspectives

    .

     I

     enjoy

     learning

     new

     skills

     and

     technologies

    ,

     and

     I

    'm

     always

     looking

     for

     new

     challenges

     to

     challenge

     myself

    .

     I

    'm

     a

     friendly

     and

     approach

    able

     person

     who

     loves

     to

     share

     my

     knowledge

     and

     experiences

     with

     others

    .

     Let

     me

     know

     if

     you

    'd

     like

     to

     meet

     me

     or

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     of

     France

     and

     serves

     as

     the

     largest

     city

     in

     the

     country

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     vibrant

     culture

    .

     The

     city

     is

     home

     to

     many

     famous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     a

     center

     for

     fashion

    ,

     art

    ,

     and

     cinema

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     a

     cultural

     hub

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

     known

     for

     its

     diverse

     population

     and

     its

     emphasis

     on

     cultural

     diversity

    .

     
    


    Paris

     is

     the

     seat

     of

     the

     French

     government

    ,

     the

     European

     Parliament

    ,

     and

     the

     French

     Senate

    .

     It

     is

     also

     home

     to

     the

     offices

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     machine

     learning

     and

     algorithms

    ,

     changes

     in

     the

     way

     data

     is

     collected

     and

     analyzed

    ,

     and

     shifts

     in

     the

     demand

     for

     AI

     technology

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     There

     is

     a

     growing

     awareness

     of

     the

     ethical

     implications

     of

     AI

     technology

    ,

     particularly

     in

     terms

     of

     privacy

    ,

     bias

    ,

     and

     human

    -machine

     interaction

    .

     AI

     systems

     will

     be

     required

     to

     balance

     the

     need

     for

     efficiency

     and

     innovation

     with

     considerations

     of

     fairness

     and

     responsibility

    .
    


    2

    .

     Development

     of

     AI

     for

     personalized

     medicine

    :

     AI

     will

     play

     a

     significant

     role

     in

     improving

     the

     accuracy

     and

     efficiency

     of

     personalized

     medicine

    ,

     allowing

     for

     the

     development

    



```python
llm.shutdown()
```
