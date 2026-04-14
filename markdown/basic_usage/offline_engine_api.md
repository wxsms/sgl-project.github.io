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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.77it/s]


    2026-04-14 20:17:32,807 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 20:17:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.83it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.83it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.83it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.83it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.83it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.83it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.83it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.83it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.83it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.83it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 27.93it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 27.93it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 27.93it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 27.93it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 27.93it/s]

    Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 27.93it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 27.93it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 27.93it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 34.24it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 34.24it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 34.24it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 34.24it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 34.24it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 34.24it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 34.24it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 34.24it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 40.13it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 40.13it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 40.13it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 40.13it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 40.13it/s]

    Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 40.13it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 40.13it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 40.13it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 40.13it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 40.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.68 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.36it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.36it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.36it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.36it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.36it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.36it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.51it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.51it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.51it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.51it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.51it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.51it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.51it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.32it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.32it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.32it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.32it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.32it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.32it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.32it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.93it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.93it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.93it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.93it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.93it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.93it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.93it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.20it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.20it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.20it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.20it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.20it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.20it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.20it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.64it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.64it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.64it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.64it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.64it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.64it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.64it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.19it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.19it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.19it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 43.72it/s]


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
    Generated text:  Kat. I'm a student of sixth grade, I'm very clever, and I'm really good at math. However, I have some problems in social skills. For example, I can't stand other people talking or being criticized. I want to improve my social skills, so please give me some advice on how to improve them. In addition, my parents are very strict with me. They are worried that I will make friends with bad people. I have some serious doubts about my future career. What should I do?
    A. Have a talk with my parents about it and find out the truth.
    B. Do more homework and improve
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the legislative branch. What is the legislative branch?
    The legislative branch of the United States government is the United States Congress. It consists of two chambers: the House of Representatives and the Senate. Each chamber has its own unique structure and responsibilities, and the power and functions of each chamber are defined by the Constitution. The House of Representatives is the lower house, while the Senate is the upper house. Together, these two chambers make up the Congress and are responsible for representing the United States in the federal government. The President is a member of the executive branch, which is responsible for implementing the laws passed by Congress and managing the federal
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The provincial capital of the same city is Lyon. Both are located in the Central Park area of the city. Which of the following statements about the capital of France is true? 
    A: The capital is Paris
    B: Lyon is a provincial capital
    C: The capital of France is located in the western part of the city
    D: The provincial capital of the capital of France is Lyon
    To determine which statement about the capital of France is true, let's analyze each option step by step:
    
    1. **Identify the capital of France:**
       - The capital of France is indeed Paris.
    
    2. **Ident
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting. Here’s what is already happening, and what will happen next. 
    
    The fast pace of AI development means that it seems impossible to stay on top of it all. But, in a world where life is increasingly complex, AI is becoming more and more important. It’s like a house of cards, and the crumple of one side can cause the whole structure to fall.
    
    One of the primary ways AI is being applied is in the healthcare sector. Here, AI is being used to help diagnose and treat diseases and injuries. It is also being used in manufacturing, financial services, transportation, and more.
    
    In the field


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I'm always looking for ways to improve my skills and knowledge. I'm also a [job title] at [company name], and I'm always eager to learn and grow. What's your favorite hobby or activity? I'm a [job title] at [company name], and I love [favorite hobby or activity]. I'm always looking for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that serves as the political, cultural, and economic center of the country. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its rich history, including the French Revolution and the French Revolution Museum. The city is home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is also known for its delicious cuisine, including French cuisine and international cuisine. The city is also home to many cultural events and festivals throughout the year, including the Eiffel Tower Festival and the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some possible future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and productive machines that can perform tasks that were previously done by humans.
    
    2. Enhanced human-computer interaction: AI is likely to become more integrated into our daily lives, allowing humans to interact with machines in a more natural and intuitive way. This could lead to a more seamless
    


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
    Generated text:  [Name] and I'm a [job title] in the [field of expertise] field. I have [number of years of experience] of experience in [specific skills or areas of expertise]. I enjoy [reason for my interest in the field] and I strive to [how I aim to contribute to the organization's success]. I am a [type of person], [what kind of personality traits I have]. I am [what I like to do for entertainment]. I am [who I would like to become as a future professional]. I have [number of books, articles, or other resources] that I have read in my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    This statement encapsulates the geographical and political importance of Paris as the capital city of France, focusing on its status as the country's capital and the main metropolis. However, if you're looking for a more specific or widely accepted fact, it would be:
    
    Paris is the capital city of France, located in the south of the country, on the Mediterranean coast. It is the second-largest city in France, with an estimated population of over 1 million people. Paris is considered the cultural and educational capital of Europe. It is also the world's seventh-largest metropolitan area. The city is famous for its architecture, fashion, art
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to see significant developments and innovations that will continue to shape the field. Some possible trends include:
    
    1. Improved machine learning algorithms: With the continued growth of big data, it's expected that AI will continue to rely on machine learning algorithms that can analyze and extract insights from complex data sets.
    
    2. Increased use of AI for autonomous vehicles: With the growing concern about traffic congestion and accidents, it's expected that AI will play a larger role in autonomous vehicles as they become more sophisticated and efficient.
    
    3. Improved natural language processing: With the increasing number of people using voice assistants like Siri and Alexa, it's expected that AI will continue


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

     am

     a

     [

    age

    ]

     year

     old

     [

    gender

    ]

     person

    .

     I

     have

     [

    number

     of

     pets

    ]

     pets

     who

     are

     my

     constant

     companions

    .

     I

     enjoy

     [

    job

    ,

     hobby

    ,

     or

     activity

    ]

     and

     I

     love

     to

     [

    describe

     something

     that

     makes

     you

     happy

    ].

     I

     am

     passionate

     about

     [

    describe

     your

     field

     of

     interest

     or

     expertise

    ],

     and

     I

     strive

     to

     [

    describe

     a

     goal

     or

     goal

     you

     want

     to

     achieve

    ].

     I

     believe

     that

     my

     [

    unique

     value

     or

     quality

    ]

     is

     [

    why

     it

     makes

     me

     a

     great

     fit

     for

     this

     role

    ].


    Sure

    ,

     here

    's

     a

     short

    ,

     neutral

     self

    -int

    roduction

     for

     a

     fictional

     character

    :


    "

    Hi

    ,

     my

     name

     is

     [

    Name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Does

     this

     statement

     correctly

     understand

     the

     content

     of

     the

     first

     sentence

    ?

     Yes

    ,

     it

     does

     understand

     the

     content

     of

     the

     first

     sentence

    .

     However

    ,

     it

     is

     more

     general

     than

     the

     full

     statement

    .

     Here

     is

     a

     more

     detailed

     version

    :
    


    Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     on

     the

     banks

     of

     the

     River

     Se

    ine

     in

     the

     south

     of

     the

     country

    .

     It

     is

     the

     third

     largest

     city

     in

     Europe

     by

     population

     and

     the

     second

     largest

     city

     in

     the

     world

     by

     land

     area

    .

     The

     city

     is

     renowned

     for

     its

     rich

     history

    ,

     renowned

     for

     its

     architecture

    ,

     and

     has

     been

     a

     cultural

     and

     political

     center

     since

     ancient

     times

    .

     Paris

     is

     also

     famous

     for

     its

     cuisine

     and

     fashion

    ,

     and

     it

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

    ,

     and

     it

    's

     likely

     to

     continue

     evolving

     and

     advancing

     in

     several

     key

     areas

    :
    


    1

    .

     Adv

    ancements

     in

     natural

     language

     processing

    :

     AI

     is

     already

     very

     powerful

    ,

     but

     natural

     language

     processing

     (

    N

    LP

    )

     is

     becoming

     more

     and

     more

     advanced

    .

     With

     the

     development

     of

     deep

     learning

     models

    ,

     N

    LP

     can

     perform

     more

     complex

     tasks

     like

     understanding

     context

    ,

     emotion

    ,

     and

     sarc

    asm

    ,

     and

     generating

     human

    -like

     text

    .
    


    2

    .

     More

     reliable

     AI

    :

     With

     the

     increase

     in

     data

     and

     computing

     power

    ,

     AI

     systems

     are

     becoming

     more

     robust

     and

     reliable

    .

     However

    ,

     there

     are

     still

     some

     risks

     associated

     with

     AI

    ,

     such

     as

     privacy

     concerns

     and

     the

     risk

     of

     algorithm

    ic

     bias

    .
    


    3

    .

     Integration

     with

    



```python
llm.shutdown()
```
