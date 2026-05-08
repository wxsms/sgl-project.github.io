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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.61it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.59it/s]


    2026-05-08 18:57:01,017 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 18:57:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:44,  3.94s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:44,  3.94s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.85it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.85it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.79it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.25it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.25it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.25it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.25it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.25it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.25it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.25it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.25it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.76 GB):   3%|▎         | 2/58 [00:00<00:03, 17.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.73it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=73.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.00 GB):  31%|███       | 18/58 [00:00<00:01, 36.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.99 GB):  31%|███       | 18/58 [00:00<00:01, 36.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.99 GB):  31%|███       | 18/58 [00:00<00:01, 36.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.97 GB):  31%|███       | 18/58 [00:00<00:01, 36.11it/s]Capturing num tokens (num_tokens=960 avail_mem=72.99 GB):  31%|███       | 18/58 [00:00<00:01, 36.11it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.98 GB):  31%|███       | 18/58 [00:00<00:01, 36.11it/s]Capturing num tokens (num_tokens=896 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=832 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=768 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=704 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=640 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=576 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=576 avail_mem=72.55 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=480 avail_mem=72.55 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.37it/s]

    Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 38.29it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:00<00:00, 38.29it/s]Capturing num tokens (num_tokens=320 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:00<00:00, 38.29it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 38.29it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.29it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.29it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.38it/s]

    Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=176 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=112 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.50it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=64 avail_mem=72.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.50it/s]

    Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=20 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=16 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.07it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 44.73it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 39.21it/s]


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
    Generated text:  Eunice. I live in a small town in the United States. This is my first day of school. I feel nervous, but excited at the same time. I am a first-grader and I'm happy to be here. At school, I like reading books. I also like playing with my toys. I like my teachers and classmates. I like playing with my friends. I like to draw pictures and make models. I love to sing. I love to play music. I like to watch cartoons. I like to eat vegetables and fruits. I like to listen to music. I like to play outside. I like to
    ===============================
    Prompt: The president of the United States is
    Generated text:  invited to a state dinner, but instead of a formal invitation letter, he gets a letter from his son, who is 21 years old, and asks him to sign it. The son signs it and then the daughter gets a copy of the letter. The president is 62 years old. How old is the daughter?
    To solve the problem, we need to determine the age of the daughter based on the information given about the president's age and the actions of the son and the daughter. Let's break it down step by step.
    
    1. Identify the age of the president: The president is 62 years old.
    
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Paris
    C. Paris
    D. Paris
    Answer:
    A
    
    The ultimate goal of China's economic and social development is to ___.
    A. achieve the full realization of socialism with Chinese characteristics
    B. achieve common prosperity
    C. maintain world peace and promote global development
    D. achieve the great rejuvenation of the Chinese nation
    Answer:
    D
    
    Male, 45 years old, with a history of gastric ulcer. The symptoms that typically occur in the evening and during sleep are ____. 
    A. Heartburn, nausea, and a burning sensation
    B. Joint pain, frequent
    ===============================
    Prompt: The future of AI is
    Generated text:  here. We are about to enter an era where AI is not just a tool, but an integral part of our everyday lives. AI can help us automate tasks, develop new technologies, and even help us make decisions in our own minds. However, it also raises important ethical and social concerns. In this article, we will explore some of the key ethical issues surrounding AI, including privacy, bias, and the impact of AI on employment and inequality. We will also discuss potential solutions and policies that can help address these concerns and ensure that AI is used in a responsible and beneficial way. Let's dive in and explore the fascinating world of AI


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination, attracting millions of visitors each year. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on modern French society. Paris is also home to many famous French artists, writers, and musicians. The city is a hub for business, finance, and commerce, with many international companies and institutions headquartered there.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    2. Greater integration with other technologies: AI will continue to be integrated with other technologies such as blockchain, quantum computing, and biotechnology. This will create new opportunities for AI to be used in new ways.
    
    3. Increased use of AI in healthcare: AI will be used to
    


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
    Generated text:  [Name], and I'm a [occupation] with [number of years in that occupation]. I've always been fascinated by the subject, and my passion has led me to pursue [shortly relevant question, such as "What's your favorite hobby or interest? "]. I've always been open to learning and growing, and I'm always looking for opportunities to contribute to the community. I'm always looking for new challenges and opportunities to learn and grow, and I'm eager to make a difference in the world. What can you tell me about yourself? Here's a possible self-introduction based on the provided context:
    
    "Hi,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The statement is concise and factual, providing the key details of the capital city of France. The capital is Paris, which is the most populous city in the European Union and is known for its rich cultural heritage, historical landmarks, and art scene. It is located in the heart of the Paris region and is the country's largest city. Paris is home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and numerous other attractions. Its nickname "La Ville Flottante" is derived from its floating population, with thousands of tourists daily visiting the city for its attractions. The French capital is recognized as one
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a rapidly evolving field with a wide range of potential applications and technologies. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more advanced, there will likely be increasing focus on ethical considerations such as privacy, bias, and transparency. This will require a greater emphasis on developing AI that is both transparent and accountable.
    
    2. Faster development times: With advances in machine learning and data analytics, it may become possible to develop AI systems that can be deployed quickly and with less training data than in the past. This could lead to greater efficiency and cost savings for businesses.
    
    3. More diverse use cases:


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

    ].

     I

     am

     a

     [

    occupation

    ]

     and

     I

    've

     always

     been

     [

    a

    /an

    ]

     person

    .

     I

    'm

     always

     [

    a

    /an

    ]

     with

     [

    a

    /an

    ]

     interests

    ,

     and

     I

     enjoy

     [

    a

    /an

    ]

     and

     [

    a

    /an

    ].

     I

    'm

     [

    a

    /an

    ]

     and

     I

     believe

     [

    a

    /an

    ]

     is

     [

    a

    /an

    ]

     way

     to

     live

    .

     What

    's

     your

     name

    ?

     [

    Name

    ]?

     That

    's

     great

     to

     meet

     you

    ,

     [

    Name

    ].

     I

    'm

     a

     [

    Name

    ]

     and

     I

    've

     always

     been

     [

    Name

    ]

     people

    .

     I

    'm

     always

     [

    Name

    ]

     with

     [

    Name

    ]

     interests

    ,

     and

     I

     enjoy

     [

    Name

    ]

     and

     [

    Name

    ].

     I

    'm

     [

    Name

    
    
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

     It

     is

     home

     to

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     famous

     for

     its

     rich

     history

    ,

     art

    ,

     and

     architecture

    .

     Paris

     has

     a

     diverse

     population

     and

     is

     a

     major

     center

     of

     business

     and

     finance

    .

     It

     is

     home

     to

     many

     important

     institutions

     and

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     The

     city

     has

     a

     rich

     cultural

     heritage

     and

     is

     known

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     social

     culture

    .

     Paris

     is

     a

     world

    -ren

    owned

     city

     that

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Love

    "

     and

     is

     a

     symbol

     of

     the

     French

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     see

     significant

     advancements

     in

     several

     areas

    ,

     including

    :
    


    1

    .

     Improved

     precision

     and

     accuracy

    :

     AI

     is

     constantly

     improving

     its

     ability

     to

     process

     and

     analyze

     data

    ,

     leading

     to

     more

     precise

     and

     accurate

     predictions

    ,

     diagnoses

    ,

     and

     decisions

    .
    


    2

    .

     Personal

    ization

     and

     adapt

    ability

    :

     AI

     is

     expected

     to

     become

     more

     personalized

    ,

     allowing

     machines

     to

     learn

     from

     their

     interactions

     and

     adapt

     to

     new

     situations

    .
    


    3

    .

     Eth

    ical

     and

     legal

     challenges

    :

     As

     AI

     technology

     becomes

     more

     prevalent

    ,

     there

     will

     be

     increasing

     scrutiny

     around

     issues

     such

     as

     privacy

    ,

     bias

    ,

     and

     responsibility

    .
    


    4

    .

     Adv

    ancements

     in

     data

     and

     storage

    :

     AI

     will

     continue

     to

     rely

     on

     larger

     and

     more

     powerful

     data

     and

     storage

     systems

     to

     store

    



```python
llm.shutdown()
```
