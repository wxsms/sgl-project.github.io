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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.53it/s]


    2026-04-29 18:22:54,726 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 18:22:54] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=5632):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:30,  1.68it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:30,  1.68it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:30,  1.68it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:30,  1.68it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:30,  1.68it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:30,  1.68it/s]Compiling num tokens (num_tokens=3328):  10%|█         | 6/58 [00:04<00:30,  1.68it/s]

    Compiling num tokens (num_tokens=3072):  10%|█         | 6/58 [00:04<00:30,  1.68it/s]Compiling num tokens (num_tokens=2816):  10%|█         | 6/58 [00:04<00:30,  1.68it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:04<00:09,  4.83it/s] Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03,  9.43it/s]

    Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.04it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.04it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.04it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.04it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.04it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.04it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.04it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.04it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.04it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 22.04it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 29.66it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 29.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.22 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.22 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.22 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.22 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.22 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.19 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.18 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.16 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.16 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.48it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.14 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.14 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.19it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=116.13 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.10 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=960 avail_mem=116.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.83it/s] Capturing num tokens (num_tokens=896 avail_mem=116.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=832 avail_mem=116.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.83it/s]

    Capturing num tokens (num_tokens=768 avail_mem=116.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=768 avail_mem=116.10 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.50it/s]Capturing num tokens (num_tokens=704 avail_mem=116.10 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.50it/s]Capturing num tokens (num_tokens=640 avail_mem=116.10 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.50it/s]Capturing num tokens (num_tokens=576 avail_mem=116.10 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.50it/s]Capturing num tokens (num_tokens=512 avail_mem=116.08 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.50it/s]Capturing num tokens (num_tokens=512 avail_mem=116.08 GB):  50%|█████     | 29/58 [00:01<00:00, 30.85it/s]Capturing num tokens (num_tokens=480 avail_mem=116.10 GB):  50%|█████     | 29/58 [00:01<00:00, 30.85it/s]Capturing num tokens (num_tokens=448 avail_mem=116.10 GB):  50%|█████     | 29/58 [00:01<00:00, 30.85it/s]

    Capturing num tokens (num_tokens=416 avail_mem=116.09 GB):  50%|█████     | 29/58 [00:01<00:00, 30.85it/s]Capturing num tokens (num_tokens=384 avail_mem=116.09 GB):  50%|█████     | 29/58 [00:01<00:00, 30.85it/s]Capturing num tokens (num_tokens=384 avail_mem=116.09 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=352 avail_mem=116.09 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=320 avail_mem=116.08 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=288 avail_mem=116.08 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=256 avail_mem=116.08 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=240 avail_mem=116.07 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=240 avail_mem=116.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=224 avail_mem=116.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=208 avail_mem=116.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.71it/s]

    Capturing num tokens (num_tokens=192 avail_mem=116.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=176 avail_mem=116.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.71it/s]Capturing num tokens (num_tokens=176 avail_mem=116.06 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.62it/s]Capturing num tokens (num_tokens=160 avail_mem=116.06 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.62it/s]Capturing num tokens (num_tokens=144 avail_mem=116.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.62it/s]Capturing num tokens (num_tokens=128 avail_mem=116.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.62it/s]Capturing num tokens (num_tokens=112 avail_mem=116.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.62it/s]Capturing num tokens (num_tokens=112 avail_mem=116.05 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.55it/s]Capturing num tokens (num_tokens=96 avail_mem=116.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.55it/s] Capturing num tokens (num_tokens=80 avail_mem=116.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.55it/s]

    Capturing num tokens (num_tokens=64 avail_mem=116.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.55it/s]Capturing num tokens (num_tokens=48 avail_mem=116.03 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.55it/s]Capturing num tokens (num_tokens=48 avail_mem=116.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=32 avail_mem=116.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=28 avail_mem=116.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=24 avail_mem=116.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=20 avail_mem=116.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=20 avail_mem=116.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.87it/s]Capturing num tokens (num_tokens=16 avail_mem=116.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.87it/s]

    Capturing num tokens (num_tokens=12 avail_mem=116.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.87it/s]Capturing num tokens (num_tokens=8 avail_mem=116.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.87it/s] Capturing num tokens (num_tokens=4 avail_mem=116.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 34.87it/s]Capturing num tokens (num_tokens=4 avail_mem=116.01 GB): 100%|██████████| 58/58 [00:01<00:00, 33.98it/s]Capturing num tokens (num_tokens=4 avail_mem=116.01 GB): 100%|██████████| 58/58 [00:01<00:00, 30.94it/s]


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
    Generated text:  Samantha and I'm an English teacher at a kindergarten. I've been teaching for the past 8 years and I love being in the classroom! In my current classroom, I am working with a class of 20 children. Our main goal is to teach them new words and provide them with extra practice to improve their language skills. I'm planning a themed week where we will read a book on a specific day of the week, then have a discussion on the week's themes and the week's main characters, and finally, a fun activity to help them practice their speaking skills. My students are not only learning how to read and write
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet tall. If a statue of the United States, which stands 30 feet tall, is located between the president and the president-elect, who is 1 foot shorter than the president, how tall is the statue relative to the president-elect? To determine the height of the statue relative to the president-elect, we need to follow these steps:
    
    1. Identify the height of the president of the United States.
    2. Identify the height of the president-elect.
    3. Determine how tall the statue of the United States is relative to the president-elect.
    
    First, the height of the president of the United States is given as 
    ===============================
    Prompt: The capital of France is
    Generated text:  _________.____
    A. Paris
    B. Lille
    C. London
    D. Brussels
    Answer:
    A
    
    The common prosperity of the whole society refers to ( ).
    A. Achieving common prosperity among all citizens
    B. Achieving common prosperity among all workers
    C. Achieving common prosperity among all rural households
    D. Achieving common prosperity among all people
    Answer:
    A
    
    There are four classes of medical ethics, which are ( ).
    A. Subjective ethics, utilitarianism, objectivism, and relativism
    B. Subjective ethics, utilitarianism, objectivism, and utilitarianism
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to be guided by 10 key trends, according to a new research report from the Future of Life Institute.
    AI will play a pivotal role in society, impacting everything from transportation to healthcare, from water management to education.
    The research includes key areas to watch, including how to deal with AI bias, how to ensure security and privacy, and the need to secure the world’s data.
    It also looks at how to ensure that AI can be used to benefit society and not harm it.
    The report, “In the Age of AI: What to Watch for in 2024,” is a new report from the Future of


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm [age] years old, and I have [number] years of experience in [job title]. I'm a [job title] at [company name], and I'm always looking for ways to [describe your job role]. I'm a [job title] at [company name], and I'm always looking for ways to [describe your job role]. I'm a [job title] at [company name], and I'm always looking for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major center for international business and diplomacy. The city is home to many important historical and cultural sites, including the Louvre, the Notre-Dame Cathedral, and the Champs-Élysées. Paris is a vibrant and diverse city with a rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve efficiency. As AI technology continues to advance, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud detection, and trading algorithms. As AI technology
    


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
    Generated text:  [insert first name]. I'm 25 years old, and I work as a [insert job title]. I'm passionate about [insert personal passion or hobby], and I enjoy [insert hobbies or interests]. I'm always looking for ways to [insert activity or project that I'm interested in]. And I'm also a [insert hobby or interest that I enjoy], which has helped me [insert impact on my life or growth]. I have a deep appreciation for the value of [insert something specific], and I believe that [insert personal belief or opinion]. What kind of person are you? I'm [insert personality trait or characteristic
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, famous landmarks, and diverse cuisine. It serves as the political, cultural, and economic center of the country and is an international tourist destination. The city's UNESCO World Heritage site includes iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also home to many world-renowned art, music, and fashion scenes. The city has a well-developed transportation network, including the famous metro and bus systems, making it convenient for residents and tourists alike. Paris has a rich cultural heritage and continues to be a global center of art, science, and technology
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and constantly evolving, with many potential directions and potential challenges.
    
    One trend that is currently gaining momentum is the development of AI that can achieve true artificial general intelligence (AGI), the ability to understand, reason, and learn in multiple domains without any specific programming. This is currently being explored in areas such as self-driving cars, natural language processing, and image recognition.
    
    Another trend is the increasing reliance on AI in areas such as healthcare, finance, and government, where the ability to quickly analyze large amounts of data and make predictions or recommendations can have a significant impact on the quality of outcomes. This is also leading to the development of more


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

    ’m

     a

     [

    Title

    ]

     at

     [

    Company

    /

    organization

    ].

     I

    ’m

     a

     [

    job

     title

    ]

     with

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     [

    field

    ].

     [

    Name

    ]

     is

     always

     looking

     to

     grow

     and

     learn

    .

     I

    ’m

     passionate

     about

     [

    job

     title

    ],

     and

     I

    ’m

     constantly

     learning

     new

     things

    ,

     learning

     from

     my

     colleagues

    ,

     and

     learning

     from

     my

     clients

    .

     I

    ’m

     a

     team

     player

    ,

     and

     I

    ’m

     always

     willing

     to

     help

     others

    .

     [

    Name

    ]

     is

     a

     [

    job

     title

    ]

     who

     is

     always

     looking

     to

     improve

    ,

     and

     I

    ’m

     always

     looking

     for

     new

     challenges

     to

     meet

    .

     I

    ’m

     always

     available

     to

     help

     anyone

     who

     needs

     a

     hand

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     is

     concise

     because

     it

     provides

     the

     essential

     information

     about

     the

     capital

     city

     in

     one

     sentence

    .

     The

     statement

     is

     factual

     as

     it

     accurately

     describes

     the

     capital

     city

     of

     France

    .

     It

     also

     doesn

    't

     contain

     any

     assumptions

     or

     add

    ictions

    ,

     leaving

     the

     reader

     with

     a

     clear

     and

     straightforward

     understanding

     of

     the

     city

    's

     location

    .

     Additionally

    ,

     the

     statement

     is

     brief

    ,

     making

     it

     easily

     digest

    ible

     for

     readers

     who

     might

     not

     be

     familiar

     with

     the

     capital

     city

    .

     
    


    In

     conclusion

    ,

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     and

     it

     is

     the

     largest

     city

     in

     the

     country

     by

     population

    ,

     with

     over

     

    2

    7

     million

     inhabitants

    .

     It

     is

     known

     for

     its

     rich

     cultural

     heritage

    ,

     diverse

     street

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     driven

     by

     several

     trends

    ,

     including

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     diagnose

     and

     treat

     diseases

    ,

     and

     the

     technology

     is

     expected

     to

     continue

     to

     improve

     in

     the

     coming

     years

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

     (

    N

    LP

    )

     will

     continue

     to

     become

     more

     advanced

    ,

     allowing

     AI

     systems

     to

     understand

     human

     language

     and

     interact

     with

     humans

     in

     new

     ways

    .
    


    3

    .

     Improved

     security

    :

     AI

     systems

     will

     continue

     to

     be

     developed

     to

     protect

     against

     cyber

     attacks

     and

     other

     types

     of

     security

     threats

    .
    


    4

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     be

     integrated

     with

     other

     technologies

     such

     as

     the

    



```python
llm.shutdown()
```
