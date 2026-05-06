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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.79it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.78it/s]


    2026-05-06 00:57:32,154 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 00:57:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.32it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.53it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.55it/s]

    Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.55it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.55it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 35.77it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 35.77it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 35.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.99it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.24it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.14it/s]Capturing num tokens (num_tokens=224 avail_mem=74.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.14it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.14it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.14it/s]

    Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.14it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.14it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.52it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.13it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.48it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.48it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.48it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.48it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.48it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.48it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 44.36it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 39.44it/s]


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
    Generated text:  Amy and I’m a filmmaker from Seattle, Washington. I’ve been making films since 1982, and I’m a professional videographer and a producer.
    After a career spanning over 40 years, my film career is now in its infancy. In fact, I’m still making films, but the way I make them and the way I want to make them has evolved and changed in the last 15 years or so. My aim is to make stories for audiences by making stories.
    I’ve always been interested in making documentaries that tell stories that need to be told. The types of stories I’m interested in include
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. He needs 57 votes to win. If he has already received 70 votes, how many more votes does he need to win? The president needs 57 votes to win.
    He has already received 70 votes.
    To determine how many more votes he needs, we subtract the number of votes he has already received from the total number of votes required.
    57 - 70 = -3
    Therefore, the president does not need any more votes to win.
    #### -3
    The answer is: -3
    ===============================
    Prompt: The capital of France is
    Generated text:  _____.
    A. Paris
    B. New Orleans
    C. Lisbon
    D. Geneva
    Answer:
    A
    
    Which of the following is NOT a characteristic of the Internet of Things (IoT)?
    A. Sharing
    B. Communication
    C. Integration
    D. Automation
    Answer:
    D
    
    "Social harmony and stability" falls under which category of the basic content of "Building a moderately prosperous society in all respects"?
    A. Goal
    B. Task
    C. Goal and Task
    D. Requirement
    Answer:
    C
    
    The philosophical concept that represents a unity of opposites, reflecting the movement of the essence and development of
    ===============================
    Prompt: The future of AI is
    Generated text:  here
    
    As an AI specialist and developer, I have been a fan of artificial intelligence for a long time. I was first introduced to artificial intelligence when I first read about it, but I never had the chance to fully understand it. I am a huge believer in the power of AI. It can transform our world, and it is doing so in many other ways. Here is what I have learned about AI in the last few years, and how it is transforming the way we live and work.
    
    The future of AI is here
    
    First of all, I have become a fan of AI. I would like to believe that there is


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and opera houses. Paris is a popular tourist destination, known for its rich history, vibrant nightlife, and diverse food scene. It is also home to numerous museums, theaters, and opera houses, including the Opéra Garnier and the Théâtre des Champs-Élysées. Paris is a city of contrasts, with its historic architecture and modern art, and is a major hub for global affairs and diplomacy
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations and guidelines for its development and use. This could lead to more robust and transparent AI systems
    


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
    Generated text:  ___________ and I'm a/an ___________ person. I'm ___________ years old, and I have a/an ___________ personality.
    
    I'm a/an ___________ individual. I'm passionate about ___________ and often try to inspire and motivate others to do the same. I believe that ___________ is the key to a successful life.
    
    I'm a/an ___________ person. I have a/an ___________ attitude towards life. I'm always willing to put in the effort and take risks to achieve my goals. I value ___________ and think that ___________ is the foundation of any successful
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its rich cultural heritage, stunning architecture, and vibrant nightlife.
    Paris is the capital city of France, renowned for its rich cultural heritage, stunning architecture, and vibrant nightlife. It is home to many iconic landmarks such as the Eiffel Tower and the Louvre Museum, and is also a major cultural hub for France and the world. The city's nightlife is highly rated, with numerous bars, restaurants, and clubs to choose from, and it is a popular destination for tourists and locals alike. Paris is also known for its annual couture and fashion week, and its historical and artistic landmarks, including the Arc
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and rapidly evolving. Here are some possible trends that we can expect to see in the AI field in the next decade:
    
    1. Increased accuracy and precision: AI will continue to improve its ability to process and analyze large amounts of data. This will result in more accurate and precise predictions, diagnoses, and recommendations.
    
    2. Increased autonomy and self-learning: AI will become more capable of making decisions and taking actions on its own without human intervention. This will allow AI systems to become more autonomous and self-learning, which will lead to more efficient and effective use of resources.
    
    3. Increased integration with other technologies: AI will become more integrated with


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

    ]

     and

     I

     am

     a

     [

    Your

     Profession

    ]

     with

     over

     [

    Number

    ]

     years

     of

     experience

     in

     the

     field

    .

     I

     enjoy

     [

    Your

     Passion

    ]

     and

     I

     am

     always

     looking

     to

     improve

     myself

     in

     order

     to

     stay

     current

     with

     the

     latest

     advancements

     in

     my

     field

    .

     I

     am

     a

     [

    Your

     Character

    ]

     and

     I

     am

     always

     willing

     to

     learn

     and

     adapt

     to

     new

     challenges

    .

     I

     am

     passionate

     about

     [

    Your

     Passion

    ]

     and

     I

     am

     confident

     in

     my

     ability

     to

     contribute

     to

     the

     success

     of

     [

    Your

     Company

    ]

     and

     I

     am

     always

     eager

     to

     bring

     my

     skills

     to

     the

     table

    .

     Please

     let

     me

     know

     if

     you

     would

     like

     me

     to

     elaborate

     on

     any

     of

     this

     or

     if

     you

     would

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

     and

     its

     capital

    .

     The

     city

     is

     located

     on

     the

     River

     Se

    ine

     and

     is

     the

     main

     cultural

     and

     economic

     center

     of

     France

    .

     It

     is

     known

     for

     its

     historical

     landmarks

    ,

     world

    -ren

    owned

     museums

     and

     art

     galleries

    ,

     and

     its

     rich

     culture

    .

     Paris

     has

     a

     thriving

     art

     scene

    ,

     including

     notable

     galleries

     and

     exhibitions

    ,

     and

     is

     also

     famous

     for

     its

     romantic

     cafes

     and

     bou

    lev

    ards

    .

     The

     city

     is

     also

     home

     to

     many

     world

    -f

    amous

     landmarks

    ,

     including

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

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     a

     beautiful

     and

     culturally

     rich

     city

     that

     draws

     tourists

     and

     locals

     alike

    .

     It

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     undoubtedly

     rapidly

     evolving

    ,

     with

     many

     exciting

     possibilities

     and

     challenges

     to

     consider

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

     use

     of

     AI

     in

     healthcare

    :

     AI

     will

     play

     a

     crucial

     role

     in

     the

     healthcare

     industry

    ,

     revolution

    izing

     diagnosis

    ,

     treatment

    ,

     and

     patient

     care

    .

     Machine

     learning

     algorithms

     will

     be

     used

     to

     analyze

     large

     volumes

     of

     patient

     data

    ,

     predict

     outcomes

    ,

     and

     identify

     areas

     for

     improvement

    .

     AI

     will

     also

     be

     used

     to

     develop

     personalized

     treatment

     plans

     based

     on

     individual

     patient

     characteristics

    .
    


    2

    .

     AI

     in

     finance

    :

     AI

     is

     already

     being

     used

     in

     the

     financial

     industry

    ,

     from

     fraud

     detection

     to

     trading

     algorithms

    .

     AI

     will

     continue

     to

     expand

     its

     use

     in

     finance

    ,

     with

     more

     advanced

     algorithms

    



```python
llm.shutdown()
```
