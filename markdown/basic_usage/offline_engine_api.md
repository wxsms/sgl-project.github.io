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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.21it/s]


    2026-05-13 23:28:38,002 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 23:28:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.03it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 13.52it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 13.52it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 13.52it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 13.52it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 13.52it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.52it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.52it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.52it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.52it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 19.63it/s]

    Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:05<00:00, 28.89it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 40.25it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 40.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.76 GB):   3%|▎         | 2/58 [00:00<00:03, 15.96it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 15.96it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 15.96it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 15.96it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.30it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.74 GB):   9%|▊         | 5/58 [00:00<00:02, 20.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.73 GB):   9%|▊         | 5/58 [00:00<00:02, 20.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.15it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=61.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.72 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.72 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.72 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.71 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.71 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.70 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.77it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.77it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=61.68 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.77it/s]Capturing num tokens (num_tokens=960 avail_mem=61.69 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.77it/s] Capturing num tokens (num_tokens=960 avail_mem=61.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.68it/s]Capturing num tokens (num_tokens=896 avail_mem=61.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.68it/s]Capturing num tokens (num_tokens=832 avail_mem=61.68 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.68it/s]Capturing num tokens (num_tokens=768 avail_mem=61.68 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.68it/s]Capturing num tokens (num_tokens=704 avail_mem=61.68 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.68it/s]Capturing num tokens (num_tokens=640 avail_mem=61.67 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.68it/s]

    Capturing num tokens (num_tokens=640 avail_mem=61.67 GB):  47%|████▋     | 27/58 [00:01<00:01, 19.44it/s]Capturing num tokens (num_tokens=576 avail_mem=61.67 GB):  47%|████▋     | 27/58 [00:01<00:01, 19.44it/s]Capturing num tokens (num_tokens=512 avail_mem=61.66 GB):  47%|████▋     | 27/58 [00:01<00:01, 19.44it/s]Capturing num tokens (num_tokens=480 avail_mem=61.67 GB):  47%|████▋     | 27/58 [00:01<00:01, 19.44it/s]Capturing num tokens (num_tokens=448 avail_mem=61.67 GB):  47%|████▋     | 27/58 [00:01<00:01, 19.44it/s]Capturing num tokens (num_tokens=416 avail_mem=61.67 GB):  47%|████▋     | 27/58 [00:01<00:01, 19.44it/s]Capturing num tokens (num_tokens=416 avail_mem=61.67 GB):  55%|█████▌    | 32/58 [00:01<00:01, 24.25it/s]Capturing num tokens (num_tokens=384 avail_mem=61.67 GB):  55%|█████▌    | 32/58 [00:01<00:01, 24.25it/s]Capturing num tokens (num_tokens=352 avail_mem=61.66 GB):  55%|█████▌    | 32/58 [00:01<00:01, 24.25it/s]Capturing num tokens (num_tokens=320 avail_mem=61.66 GB):  55%|█████▌    | 32/58 [00:01<00:01, 24.25it/s]Capturing num tokens (num_tokens=288 avail_mem=61.66 GB):  55%|█████▌    | 32/58 [00:01<00:01, 24.25it/s]Capturing num tokens (num_tokens=256 avail_mem=61.65 GB):  55%|█████▌    | 32/58 [00:01<00:01, 24.25it/s]

    Capturing num tokens (num_tokens=256 avail_mem=61.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 28.79it/s]Capturing num tokens (num_tokens=240 avail_mem=61.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 28.79it/s]Capturing num tokens (num_tokens=224 avail_mem=61.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 28.79it/s]Capturing num tokens (num_tokens=208 avail_mem=61.64 GB):  64%|██████▍   | 37/58 [00:01<00:00, 28.79it/s]Capturing num tokens (num_tokens=192 avail_mem=61.64 GB):  64%|██████▍   | 37/58 [00:01<00:00, 28.79it/s]Capturing num tokens (num_tokens=176 avail_mem=61.64 GB):  64%|██████▍   | 37/58 [00:01<00:00, 28.79it/s]Capturing num tokens (num_tokens=176 avail_mem=61.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=160 avail_mem=61.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=144 avail_mem=61.63 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=128 avail_mem=61.63 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=112 avail_mem=61.63 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=96 avail_mem=61.62 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.04it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=61.62 GB):  81%|████████  | 47/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=80 avail_mem=61.62 GB):  81%|████████  | 47/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=64 avail_mem=61.62 GB):  81%|████████  | 47/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=48 avail_mem=61.61 GB):  81%|████████  | 47/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=32 avail_mem=61.61 GB):  81%|████████  | 47/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=28 avail_mem=61.60 GB):  81%|████████  | 47/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=28 avail_mem=61.60 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.40it/s]Capturing num tokens (num_tokens=24 avail_mem=61.60 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.40it/s]Capturing num tokens (num_tokens=20 avail_mem=61.60 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.40it/s]Capturing num tokens (num_tokens=16 avail_mem=60.23 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.40it/s]Capturing num tokens (num_tokens=12 avail_mem=60.23 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.40it/s]Capturing num tokens (num_tokens=8 avail_mem=60.23 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.40it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=60.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=4 avail_mem=60.22 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=4 avail_mem=60.22 GB): 100%|██████████| 58/58 [00:01<00:00, 31.96it/s]


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
    Generated text:  Jack Chen, a 21-year-old college student from Shanghai. I have just finished a college-level mathematics course, and I have a lot of problems to solve. Can you give me some advice on how to improve my problem-solving skills?
    
    Certainly! Here are some tips that can help improve your problem-solving skills:
    
    1. **Understand the Problem**: Before starting to solve, make sure you understand the problem statement. Break it down into smaller parts and make sure you have all the necessary information.
    
    2. **Identify the Goal**: Clearly define what you want to achieve with the problem. This will guide your thinking and help you
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to finalize plans for a new budget for the upcoming fiscal year. He has 100 senators and 50 representatives. Each senator can vote for 1 of 10 options, and each representative can vote for 1 of 5 options. If there are 150 candidates who are eligible for a first round of voting, how many more votes are needed for each option? To determine how many more votes are needed for each option, we need to follow these steps:
    
    1. Calculate the total number of votes each senator can cast.
    2. Calculate the total number of votes each representative can cast.
    3.
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Berlin
    C. Moscow
    D. Los Angeles
    Answer: A
    
    Which of the following statements about the steam turbine speed control system is incorrect?
    A. It is composed of a speed control system and a speed feedback system.
    B. The speed control system is responsible for adjusting the speed of the steam turbine, while the speed feedback system adjusts the steam pressure.
    C. When the steam turbine speed exceeds the rated speed, the turbine will automatically stop running.
    D. In a steam turbine speed control system, there is usually a steam turbine governor that automatically adjusts the steam pressure.
    Answer: D
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  here. But what are the necessary steps to get there? Here’s a look at the key aspects of the future of AI.
    The future of AI is here. But what are the necessary steps to get there? Here’s a look at the key aspects of the future of AI.
    The future of AI is here. But what are the necessary steps to get there? Here’s a look at the key aspects of the future of AI.
    The future of AI is here. But what are the necessary steps to get there? Here’s a look at the key aspects of the future of AI.
    The future of AI is here. But what


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new experiences and learning opportunities. What do you like to do for fun? I love [insert a short description of your favorite hobby or activity]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite thing about being a [job title] at [company name]?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French language. It is also a popular tourist destination, with millions of visitors annually. The city is home to many notable French artists, writers, and musicians, and is known for its cuisine, including its famous croissants and pastries. Paris is a vibrant and dynamic city, with a rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in even more areas, including diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection, risk assessment, and trading strategies. As AI becomes more advanced, it is
    


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
    Generated text:  [Name]. I'm a [job title] with [role], located in [location]. I enjoy [what I like to do]. And I really like [why I enjoy what I enjoy]. And, as a [skill or hobby], I really like [what I like]. My [add-on] personality trait is [what it is]. And I really enjoy [why I enjoy that]. So, thank you for asking, I look forward to meeting you! 🎓💼💼💡🎨 
      
     Note: You can choose any of the options listed above and rephrase them in a more neutral and friendly tone. Let me know
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city and the capital of France, located in the South Eastern Region of France. It is the seat of the French government and is home to the headquarters of France’s most famous monuments and landmarks. Paris is a blend of the traditional and the modern with its rich and diverse history, and is known for its culture, cuisine, fashion, and architecture. It is a city of contrasts, with its towering buildings, museums, and historic neighborhoods, and it is a popular tourist destination and cultural center. Paris is also a symbol of French identity and history. 
    
    French President Emmanuel Macron is the current leader of the country
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a variety of trends and developments that are currently emerging. Some of the most promising trends include:
    
    1. Increased integration with human perception: As AI becomes more capable of processing visual and sensory information, it may be able to learn and adapt to complex visual patterns and objects, opening up new possibilities for image recognition, object recognition, and more.
    
    2. Development of AI that can learn and improve on its own: AI systems that are able to learn and improve on their own may be able to adapt to changing conditions and improve their performance over time.
    
    3. Greater focus on ethical considerations: As AI becomes more integrated into


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

     come

     from

     [

    Name

    ],

     a

     place

     that

     has

     seen

     many

     changes

     and

     tumult

    uous

     times

     throughout

     its

     history

    .

     I

     hold

     a

     deep

     connection

     to

     this

     land

     and

     its

     people

    ,

     and

     I

     am

     constantly

     striving

     to

     learn

     more

     about

     its

     rich

     cultural

     heritage

     and

     history

    .

     I

     am

     also

     passionate

     about

     preserving

     and

     promoting

     our

     traditions

    ,

     and

     I

     am

     determined

     to

     ensure

     that

     they

     continue

     to

     thrive

     for

     generations

     to

     come

    .

     Let

     me

     know

     if

     you

     would

     like

     to

     speak

     with

     me

     about

     anything

    ,

     and

     I

     will

     be

     happy

     to

     chat

     and

     learn

     from

     you

    .

     Looking

     forward

     to

     our

     conversation

    !

     [

    Name

    ]

     Let

     me

     know

     if

     you

     have

     any

     questions

    ,

     and

     I

     will

     be

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     country

     and

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

     cultural

     scene

    .

     Paris

     is

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     many

     other

     iconic

     landmarks

     that

     showcase

     its

     artistic

     and

     historical

     heritage

    .

     The

     city

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     with

     iconic

     fashion

     houses

     like

     Chanel

     and

     Louis

     V

    uit

    ton

    ,

     and

     its

     cosm

    opolitan

     atmosphere

    .

     Paris

     is

     a

     bustling

     met

    ropolis

     with

     a

     population

     of

     over

     a

     million

     people

    .

     It

     is

     a

     major

     cultural

     hub

     that

     plays

     an

     important

     role

     in

     France

    's

     political

     and

     economic

     life

    .

     Paris

     is

     often

     referred

     to

     as

     the

     "

    city

     of

     a

     thousand

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     potential

    .

     Here

     are

     some

     of

     the

     most

     promising

     trends

    :
    


    1

    .

     Increased

     AI

     privacy

    :

     As

     AI

     systems

     become

     more

     sophisticated

     and

     complex

    ,

     there

     will

     be

     increased

     concerns

     about

     the

     privacy

     and

     security

     of

     personal

     data

    .

     Developers

     and

     users

     will

     need

     to

     find

     ways

     to

     protect

     personal

     information

     and

     ensure

     that

     AI

     systems

     are

     used

     eth

    ically

    .
    


    2

    .

     AI

     autonomy

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     there

     will

     be

     a

     growing

     trend

     toward

     developing

     autonomous

     AI

     systems

     that

     can

     operate

     independently

     and

     make

     decisions

     without

     human

     intervention

    .
    


    3

    .

     AI

     ethics

    :

     AI

     is

     already

     starting

     to

     make

     headlines

     with

     issues

     like

     bias

     and

     discrimination

    ,

     but

     as

     AI

     systems

     become

     more

     complex

     and

    



```python
llm.shutdown()
```
