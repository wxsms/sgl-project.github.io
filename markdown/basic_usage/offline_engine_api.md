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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.91it/s]


    2026-05-08 23:29:18,482 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 23:29:18] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:26,  1.97it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:26,  1.97it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:26,  1.97it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:26,  1.97it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:26,  1.97it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:12,  3.99it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:12,  3.99it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:12,  3.99it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:12,  3.99it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:12,  3.99it/s]

    Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:05<00:12,  3.99it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 12.83it/s]

    Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 17.95it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 17.95it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 17.95it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 17.95it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 17.95it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 17.95it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 17.95it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:01, 23.53it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:01, 23.53it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:01, 23.53it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:01, 23.53it/s]

    Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:01, 23.53it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:01, 23.53it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:01, 23.53it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 34.22it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 34.22it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 34.22it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 34.22it/s]

    Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 34.22it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 34.22it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 34.22it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 34.22it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 41.19it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 41.19it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 41.19it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 41.19it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 41.19it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 41.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.52 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.52 GB):   3%|▎         | 2/58 [00:00<00:04, 12.42it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.52 GB):   3%|▎         | 2/58 [00:00<00:04, 12.42it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.51 GB):   3%|▎         | 2/58 [00:00<00:04, 12.42it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.51 GB):   7%|▋         | 4/58 [00:00<00:04, 13.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.51 GB):   7%|▋         | 4/58 [00:00<00:04, 13.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.50 GB):   7%|▋         | 4/58 [00:00<00:04, 13.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.50 GB):  10%|█         | 6/58 [00:00<00:03, 14.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.48 GB):  10%|█         | 6/58 [00:00<00:03, 14.22it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=56.48 GB):  10%|█         | 6/58 [00:00<00:03, 14.22it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.48 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.47 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.46 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.45 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.45 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.32it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.43 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.32it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=56.44 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.44 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.44 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.43 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.42 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.41 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.41 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.41 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.76it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=56.13 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.13 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.76it/s]Capturing num tokens (num_tokens=960 avail_mem=55.42 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.76it/s] Capturing num tokens (num_tokens=960 avail_mem=55.42 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.52it/s]Capturing num tokens (num_tokens=896 avail_mem=55.42 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.52it/s]Capturing num tokens (num_tokens=832 avail_mem=55.41 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.52it/s]Capturing num tokens (num_tokens=768 avail_mem=55.41 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.52it/s]Capturing num tokens (num_tokens=704 avail_mem=55.41 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.52it/s]

    Capturing num tokens (num_tokens=704 avail_mem=55.41 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.05it/s]Capturing num tokens (num_tokens=640 avail_mem=55.40 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.05it/s]Capturing num tokens (num_tokens=576 avail_mem=55.40 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.05it/s]Capturing num tokens (num_tokens=512 avail_mem=55.39 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.05it/s]Capturing num tokens (num_tokens=480 avail_mem=55.40 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.05it/s]Capturing num tokens (num_tokens=480 avail_mem=55.40 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.37it/s]Capturing num tokens (num_tokens=448 avail_mem=55.40 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.37it/s]Capturing num tokens (num_tokens=416 avail_mem=55.40 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.37it/s]Capturing num tokens (num_tokens=384 avail_mem=55.40 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.37it/s]Capturing num tokens (num_tokens=352 avail_mem=55.39 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.37it/s]

    Capturing num tokens (num_tokens=320 avail_mem=55.39 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.37it/s]Capturing num tokens (num_tokens=320 avail_mem=55.39 GB):  60%|██████    | 35/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=288 avail_mem=55.39 GB):  60%|██████    | 35/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=256 avail_mem=55.38 GB):  60%|██████    | 35/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=240 avail_mem=55.38 GB):  60%|██████    | 35/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=224 avail_mem=55.38 GB):  60%|██████    | 35/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=208 avail_mem=55.37 GB):  60%|██████    | 35/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=208 avail_mem=55.37 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=192 avail_mem=55.37 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=176 avail_mem=55.37 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=160 avail_mem=55.37 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=144 avail_mem=55.36 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.73it/s]

    Capturing num tokens (num_tokens=128 avail_mem=55.36 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=128 avail_mem=55.36 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=112 avail_mem=55.36 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=96 avail_mem=55.35 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.45it/s] Capturing num tokens (num_tokens=80 avail_mem=55.35 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=64 avail_mem=55.02 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=48 avail_mem=54.92 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=48 avail_mem=54.92 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=32 avail_mem=54.91 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=28 avail_mem=54.91 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=24 avail_mem=54.91 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=20 avail_mem=54.90 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.69it/s]

    Capturing num tokens (num_tokens=16 avail_mem=54.90 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=16 avail_mem=54.90 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=12 avail_mem=54.90 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=8 avail_mem=54.89 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.31it/s] Capturing num tokens (num_tokens=4 avail_mem=54.89 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=4 avail_mem=54.89 GB): 100%|██████████| 58/58 [00:01<00:00, 30.87it/s]


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
    Generated text:  Rikke, and I'm a computer scientist with a passion for all things cybersecurity. I'm currently a junior at the University of Michigan, where I'm studying computer science and machine learning. I enjoy coding, problem-solving, and learning new things. I'm currently working on a project that involves automating the process of data mining and natural language processing. Can you provide me with some tips on how to improve my coding skills and prepare for future challenges in the field of cybersecurity? Sure, here are some tips for improving your coding skills and preparing for future challenges in the field of cybersecurity:
    
    1. Practice, practice, practice: The
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to estimate the percentage of all students in their country that have access to clean drinking water. They collect a random sample of 300 students and find that 120 of them have access to clean drinking water. What is the estimate of the percentage of all students in the United States that have access to clean drinking water?
    
    To estimate the percentage of all students in the United States that have access to clean drinking water, we can use the sample proportion and apply it to the total population of students in the country.
    
    Here are the steps to find the estimated percentage:
    
    1. **Calculate the sample proportion:**
       The sample
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. London
    C. Tokyo
    D. Shanghai
    Answer:
    
    A
    
    The party under whose leadership the Communist Party of China was established in 1921 is ____.
    A. Provisional Central Government of the Chinese Soviet Republic
    B. Provisional Central Government of the Chinese Workers' and Peasants' Red Army
    C. Provisional Central Committee of the Communist Party of China
    D. Provisional Central Government of the Chinese People's Liberation Army
    Answer:
    
    C
    
    The essence of the Chinese Dream is ____.
    A. The great rejuvenation of the Chinese nation
    B. A
    ===============================
    Prompt: The future of AI is
    Generated text:  in the clouds
    The future of AI is in the clouds
    The future of AI is in the clouds
    
    Imagine that you are a customer, and you want to learn about the future of AI. What do you want to know? How would you like to learn about it? Here are some questions and answers that could help you make your journey through the future of AI more enjoyable and informative.
    
    Q: What is the future of AI like?
    
    A: The future of AI is exciting and full of possibilities. It is predicted that by 2023, AI will account for approximately 30% of all jobs, compared to


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Person] who is [What I enjoy doing]. I'm [What I like to do]. I'm [What I'm passionate about]. I'm [What I love to do]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I enjoy doing]. I'm [What I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a popular tourist destination and a major economic center in France. It is also home to many important institutions such as the French Academy of Sciences and the French National Library. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of people, culture, and history that is a must
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be an increasing focus on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy. AI developers will need to be more mindful of the potential consequences of their work and work to ensure that it is used in a way that is fair and responsible.
    
    2. Greater integration with other technologies: AI is already being
    


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
    Generated text:  [Your Name], and I'm here to help. I'm here to assist you with any questions or concerns you might have, and I'm available 24/7. Whether you need help with anything from basic information to complex projects, I'm here to provide you with the support and guidance you need. So, if you have any questions or concerns, please feel free to ask me. Thank you. [Your Name] [Your Interests, Skills, and Areas of Expertise] I can help with a wide range of tasks and inquiries. From general information about the world to specific projects and services, I'm here to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital of France and is the largest city in the country. It has a rich history, including its origins as a Roman settlement, and is renowned for its diverse architecture, cultural offerings, and annual festivals. Paris is also a hub of commerce and finance, attracting millions of visitors each year. Its iconic landmarks, such as the Eiffel Tower and Notre-Dame Cathedral, are testament to its status as a global city. The city's elegant boulevards, and the diverse cuisine, make it a popular destination for tourists and locals alike. Paris is a major transportation hub, with the iconic metro system connecting the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve several key trends that are expected to shape the development of the technology. Here are some possible future trends in artificial intelligence:
    
    1. Increased Human-in-the-Loop: AI will continue to rely more and more on human interaction and expertise. In the future, we may see more people working alongside AI systems to improve their performance and decision-making.
    
    2. Increased Data Security: As AI systems become more complex and rely on larger amounts of data, it is likely that we will see increased attention paid to data security. This includes measures to protect against cyber threats, data breaches, and other security risks.
    
    3. AI in Healthcare:


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

    ]

     and

     I

    'm

     a

     [

    job

     title

    ]

     with

     [

    length

     of

     experience

    ]

     years

     of

     experience

     in

     [

    relevant

     field

    ].

     I

    've

     always

     been

     passionate

     about

     [

    what

     you

     enjoy

     doing

     or

     studying

    ],

     and

     I

    'm

     committed

     to

     [

    goals

     or

     objectives

     you

     have

     set

    ].

     I

     have

     a

     [

    number

    ]

     degree

     in

     [

    relevant

     degree

    ],

     and

     my

     [

    number

    ]

     years

     of

     teaching

     experience

     have

     left

     me

     with

     [

    number

    ]

     students

    .

     I

    'm

     a

     [

    yes

    /no

    ]

     person

     who

     enjoys

     [

    a

    ffect

    ing

     the

     listener

    ],

     and

     I

     believe

     in

     [

    what

     you

     would

     call

     your

     moral

     compass

    ].

     My

     [

    number

    ]

     of

     hobbies

     and

     interests

     help

     me

     stay

     [

    what

     you

     describe

     as

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    What

     is

     a

     concise

     fact

     about

     France

    's

     capital

     city

    ?
    


    Make

     it

     concise

     but

     informative

    ,

     please

    .

     Paris

     is

     the

     capital

     of

     France

    .

     It

     is

     located

     in

     the

     western

     part

     of

     the

     country

     and

     is

     the

     largest

     city

     in

     France

     by

     population

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     vibrant

     culture

    ,

     and

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     home

     to

     important

     educational

     institutions

    ,

     such

     as

     the

     É

    cole

     norm

    ale

     sup

    érie

    ure

    ,

     and

     hosts

     many

     world

    -f

    amous

     events

     and

     festivals

     throughout

     the

     year

    .

     Paris

     is

     a

     cultural

     and

     tourist

     hub

    ,

     with

     a

     diverse

     range

     of

     attractions

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

    :
    


    1

    .

     Increased

     specialization

     and

     specialization

    ization

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     is

     likely

     to

     become

     more

     specialized

     and

     targeted

     towards

     certain

     tasks

    .

     This

     may

     lead

     to

     the

     development

     of

     new

     technologies

     and

     algorithms

     that

     are

     tailored

     to

     specific

     industries

     or

     applications

    .
    


    2

    .

     Integration

     with

     human

     decision

    -making

    :

     AI

     systems

     are

     becoming

     more

     integrated

     with

     human

     decision

    -making

    ,

     with

     more

     advanced

     models

     capable

     of

     making

     decisions

     based

     on

     a

     range

     of

     inputs

     and

     feedback

     from

     humans

    .
    


    3

    .

     Personal

    ization

     and

     adapt

    ability

    :

     As

     AI

     learns

     from

     data

     and

     improves

     over

     time

    ,

     it

     is

     likely

     to

     become

     more

     personalized

     and

     adaptable

    ,

     able

     to

     learn

     and

     improve

     based

    



```python
llm.shutdown()
```
