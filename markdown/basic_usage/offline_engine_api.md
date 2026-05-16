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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.88it/s]


    2026-05-16 02:27:56,553 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 02:27:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:52,  4.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:52,  4.07s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:52,  4.07s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:52,  4.07s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:52,  4.07s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.74it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.94it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.94it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.94it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.94it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.94it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.94it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.94it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.94it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.94it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.94it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.94it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.48it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.48it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.48it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.48it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.48it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.48it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.48it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.48it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.48it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.48it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.48it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.21it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.53it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.81it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.74it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.75it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.92it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.92it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.92it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.92it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.92it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.15it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.15it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.15it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.15it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.15it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.15it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.15it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.85it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.85it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.85it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.85it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.85it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.85it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.85it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.05it/s]

    Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.05it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.05it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:01<00:00, 48.44it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 48.44it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 48.44it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 48.44it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 48.44it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 48.44it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.53it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.53it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.53it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.53it/s]

    Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.53it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.53it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.77it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.77it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 43.08it/s]


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
    Generated text:  Shu, and I'm a group project manager for the team at Autonomous Robotics & Navigation. I'm currently leading the teams in developing and improving the models and algorithms for the autonomous vehicles. My team is currently working on a new generation of self-driving cars that will have the ability to navigate through urban environments while avoiding obstacles and pedestrians. I have a strong background in robotics and artificial intelligence, and I'm passionate about pushing the boundaries of what autonomous vehicles can achieve.
    
    Could you provide me with information on the current state of autonomous driving technology and its potential future developments?
    Certainly! The field of autonomous driving technology is rapidly advancing, and there are
    ===============================
    Prompt: The president of the United States is
    Generated text:  serving his 80th birthday. He is 3 times as old as he was 30 years ago. If he was 18 years old 30 years ago, how old is he now? Let's break down the information given in the problem step by step.
    
    1. The president is 80 years old now.
    2. 30 years ago, he was 18 years old. Therefore, 30 years ago, he was \(80 - 18 = 62\) years old.
    3. According to the problem, the president's current age is 3 times his
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Nice
    C. Barcelona
    D. London
    Answer:
    A
    
    In the 1950s, the film industry had its golden age. The American film industry, led by MGM, made films like "Karbottle and the Denim Codger," "The Long Goodbye," "The Thing With Angels," and "Who Framed Roger Rabbit," which were very popular. Their style was known as the "Golden Age of American Cinema." The French film industry, on the other hand, was in the 1930s, with the influence of Hollywood in France.
    ===============================
    Prompt: The future of AI is
    Generated text:  here and it’s set to be revolutionized by the adoption of the Quantum Internet. Quantum computing is the most powerful form of computing yet, and this will enable us to solve complex problems and new areas of technology that we can’t even imagine today. With quantum computing, we will be able to solve problems that would take us centuries and billions of dollars to solve today. The exact steps to becoming a Quantum Internet user are as follows:
    1. Understanding the Quantum Internet: The Quantum Internet is a decentralized network of quantum computers that can communicate and collaborate to solve complex problems. It is based on the principles of quantum mechanics, which enable new types


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a unique skill or personality trait here]. And what's your background? I have [insert a relevant background here]. And what's your favorite hobby or activity? I enjoy [insert a hobby or activity here]. And what's your favorite book or movie? I love [insert a favorite book or movie here]. And what's your favorite color? I love [insert a favorite color here]. And what's your favorite place to go
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose". It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 5 million people. Paris is a cultural, historical, and artistic center, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and the Arc de Triomphe. It is also a major financial center and a major hub for international trade and diplomacy. Paris is a popular tourist destination and a major tourist attraction, with many museums, art galleries, and other
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and personalized interactions. This could lead to more efficient and effective use of AI in various applications, such as healthcare, finance, and education.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for the development and use of AI, as well as a greater focus on ensuring that AI systems are fair, transparent, and accountable.
    
    3. Increased use of
    


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
    Generated text:  [insert your name], and I'm [insert your profession]!
    
    I'm passionate about [insert a short description of your passion or interests]. Whether it's [insert a short reason for your passion or interest], or [insert another reason if you have any], I enjoy [insert why you love what you love].
    
    I enjoy [insert something about yourself, like hobbies, skills, or talents], and I'm always looking for new ways to develop my skills and knowledge. I'm always interested in learning more about [insert a subject you're passionate about or know something about].
    
    I'm always eager to learn and grow, and I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the city of lights, and is the most populous city in the country and the largest metropolitan area in Europe.
    
    Paris is also known as the "city of lights" because of its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as the neon-lit streets and the city's colorful culture. It is a major international financial and cultural center and is home to many of the world's top museums, art galleries, and restaurants. Paris is also an important center for politics, music, and theater, with the 2019 Summer Olympics held
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, and it is difficult to predict exactly where it will take us. However, here are some possible trends that could potentially come about in the near and long term:
    
    1. Increased automation and machine learning: As AI technology becomes more advanced, we are likely to see an increase in automation and machine learning. This could lead to the automation of many manual tasks, allowing humans to focus on more complex tasks. Additionally, AI could become more sophisticated and capable of making decisions and making choices based on complex data.
    
    2. AI integration with other technologies: AI is becoming increasingly integrated into a wide range of technologies. This integration could lead to


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

    insert

     your

     name

    ].

     I

    'm

     a

     [

    insert

     your

     occupation

     or

     professional

     field

    ]

     and

     I

     love

     to

     [

    insert

     something

     that

     makes

     you

     proud

     of

     yourself

     or

     something

     you

    're

     passionate

     about

    ].

     I

    'm

     constantly

     learning

     and

     growing

    ,

     and

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     make

     a

     difference

     in

     the

     world

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     Hello

    ,

     my

     name

     is

     [

    insert

     your

     name

    ].

     I

    'm

     a

     [

    insert

     your

     occupation

     or

     professional

     field

    ]

     and

     I

     love

     to

     [

    insert

     something

     that

     makes

     you

     proud

     of

     yourself

     or

     something

     you

    're

     passionate

     about

    ].

     I

    'm

     constantly

     learning

     and

     growing

    ,

     and

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     make

     a

     difference

     in

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Please

     answer

     the

     following

     question

     about

     this

     statement

    :
    


    How

     many

     times

     has

     Paris

     hosted

     the

     FIFA

     World

     Cup

    ?

     Paris

     has

     hosted

     the

     FIFA

     World

     Cup

     seven

     times

    .

     It

     was

     first

     held

     in

     

    1

    9

    3

    8

     and

     has

     since

     been

     held

     every

     four

     years

     since

     then

    .

     However

    ,

     the

     first

     time

     it

     was

     held

     in

     

    2

    0

    0

    2

    ,

     a

     safety

     protocol

     was

     implemented

     to

     prevent

     a

     terrorist

     attack

     during

     the

     tournament

    ,

     resulting

     in

     its

     cancellation

    .

     This

     event

     was

     later

     res

    cheduled

     to

     

    2

    0

    1

    0

    .

     Since

     then

    ,

     the

     tournament

     has

     been

     held

     in

     

    2

    0

    1

    1

    ,

     

    2

    0

    1

    4

    ,

     

    2

    0

    1

    5

    ,

     

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     with

     several

     trends

     expected

     to

     shape

     its

     landscape

    .

     Here

     are

     some

     of

     the

     most

     likely

     future

     trends

    :
    


    1

    .

     Enhanced

     AI

     capabilities

    :

     AI

     will

     continue

     to

     advance

    ,

     with

     new

     algorithms

    ,

     architectures

    ,

     and

     hardware

     designs

     emerging

     to

     improve

     its

     performance

     and

     capabilities

    .
    


    2

    .

     AI

     integration

     with

     human

     intelligence

    :

     AI

     will

     increasingly

     be

     integrated

     with

     human

     intelligence

    ,

     enabling

     humans

     to

     take

     on

     more

     complex

     tasks

     and

     collaborate

     with

     AI

     systems

    .
    


    3

    .

     Autonomous

     AI

    :

     Autonomous

     AI

     systems

     will

     become

     more

     prevalent

    ,

     with

     robots

     and

     other

     autonomous

     vehicles

     able

     to

     make

     decisions

     based

     on

     complex

     data

     and

     rules

    .
    


    4

    .

     Improved

     privacy

     and

     security

    :

     There

     will

     be

     an

     increased

     emphasis

     on

     improving

     privacy

    



```python
llm.shutdown()
```
