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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.85it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.85it/s]


    2026-05-18 17:52:34,529 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 17:52:34] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.70it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.47it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.47it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.47it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.47it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.47it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.47it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.47it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.47it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.47it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.47it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.69it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.69it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.69it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.69it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.69it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.69it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.69it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.69it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.69it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.69it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.69it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.07it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.07it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.07it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.07it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.07it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.07it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.07it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.07it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.04 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.01 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.01 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.00 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.00 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.00 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.00 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.99 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.98 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.98 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.98 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.98 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.58it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.98 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.58it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.58it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=55.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.69 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s]

    Capturing num tokens (num_tokens=960 avail_mem=54.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s] Capturing num tokens (num_tokens=896 avail_mem=54.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s]Capturing num tokens (num_tokens=832 avail_mem=54.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s]Capturing num tokens (num_tokens=832 avail_mem=54.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=768 avail_mem=54.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=704 avail_mem=54.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=640 avail_mem=54.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=576 avail_mem=54.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=512 avail_mem=54.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=512 avail_mem=54.54 GB):  50%|█████     | 29/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=480 avail_mem=54.56 GB):  50%|█████     | 29/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=448 avail_mem=54.56 GB):  50%|█████     | 29/58 [00:00<00:00, 42.04it/s]

    Capturing num tokens (num_tokens=416 avail_mem=54.55 GB):  50%|█████     | 29/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=384 avail_mem=54.55 GB):  50%|█████     | 29/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=352 avail_mem=54.55 GB):  50%|█████     | 29/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=352 avail_mem=54.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.64it/s]Capturing num tokens (num_tokens=320 avail_mem=54.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.64it/s]Capturing num tokens (num_tokens=288 avail_mem=54.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.64it/s]Capturing num tokens (num_tokens=256 avail_mem=54.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.64it/s]Capturing num tokens (num_tokens=240 avail_mem=54.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.64it/s]Capturing num tokens (num_tokens=224 avail_mem=54.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.64it/s]Capturing num tokens (num_tokens=224 avail_mem=54.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=208 avail_mem=54.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=192 avail_mem=54.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.89it/s]

    Capturing num tokens (num_tokens=176 avail_mem=54.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=160 avail_mem=54.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=144 avail_mem=54.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=144 avail_mem=54.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=128 avail_mem=54.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=112 avail_mem=54.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=96 avail_mem=54.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.22it/s] Capturing num tokens (num_tokens=80 avail_mem=54.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.22it/s]

    Capturing num tokens (num_tokens=64 avail_mem=54.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=64 avail_mem=54.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=48 avail_mem=54.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=32 avail_mem=60.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=28 avail_mem=60.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=24 avail_mem=60.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.55it/s]

    Capturing num tokens (num_tokens=24 avail_mem=60.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.05it/s]Capturing num tokens (num_tokens=20 avail_mem=60.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.05it/s]Capturing num tokens (num_tokens=16 avail_mem=60.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.05it/s]Capturing num tokens (num_tokens=12 avail_mem=60.22 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.05it/s]Capturing num tokens (num_tokens=8 avail_mem=60.22 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.05it/s] Capturing num tokens (num_tokens=4 avail_mem=60.22 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.05it/s]Capturing num tokens (num_tokens=4 avail_mem=60.22 GB): 100%|██████████| 58/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=4 avail_mem=60.22 GB): 100%|██████████| 58/58 [00:01<00:00, 36.29it/s]


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
    Generated text:  Amy. I'm from China. I am twelve years old. I can speak English. I can sing songs and dance well. I can play games with my friends. I have a big family. My mom is a teacher and she likes swimming. My dad is a driver and he likes playing football. We have a big house and we all love it. Our family likes to have a special dinner every night. They are all in a big family. My parents are not happy when my brother and I have big parties. My brother and I are very happy about this. He says he likes my family very much. I want to be
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of what type of group?
    A) The House of Representatives
    B) The Senate
    C) The Cabinet
    D) The Congress of the United States
    D) The Congress of the United States
    
    The Congress of the United States is a bicameral legislature that consists of the House of Representatives and the Senate. The president is elected by the citizens of the United States, and they serve a four-year term. The president is not a member of any specific group, but rather serves as a representative of the United States government. Therefore, the correct answer is D) The Congress of the United States.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the oldest capital in the world, and it is the most populous capital in the world. It is the seat of government, the cultural heart of the world and the main urban area of the country. The capital of France is a political, administrative, and cultural centre, located on the Île de la Cité.
    It is located in the north of the country, in the Seine river basin, and it is the second largest city in the world by area. It is situated in the central part of the region. It is the capital of the department of the Île-de-France.
    The capital of France
    ===============================
    Prompt: The future of AI is
    Generated text:  not in a matter of a few decades but in the moment, and there are a few new approaches to building up that future.
    
    The amount of data that’s needed to develop a good model for AI is massive, but there’s no single method to take in that data and use it to develop a model. This is where our team at tech startup Skymind has taken a new approach to tackle this problem.
    
    By using the techniques that Facebook’s DeepMind use, they’ve created something called the Neural Network of Mind (N-Mod). The N-Mod is a computer program that uses neural networks to create an AI that can learn


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, with many famous fashion designers and boutiques located in the city. Paris is a cultural and economic center of France and a major tourist destination. It is home to many museums, theaters, and other cultural institutions. The city is also known for its food scene, with many famous restaurants and cafes serving delicious cuisine.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced privacy and security: As AI becomes more sophisticated, there will be a need to address privacy and security concerns. This will likely involve developing new algorithms and techniques to protect against data breaches and other forms of cyber threats.
    
    3. Greater use of AI in healthcare: AI
    


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
    Generated text:  [Name] and I'm a [Age] year old [Occupation/Position]. I've always been fascinated by [Field of Interest], and I'm always on the lookout for new experiences that will push me to [A New Skill/Ability]. I'm [Any hobbies or interests] and I love [Something that makes me happy]. I'm always looking for a challenge, and I'm always eager to learn and grow. How would you describe yourself as a person? As a person, I am a curious and adventurous individual with a strong sense of self-motivation. I am always looking for new experiences and I am constantly
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is also renowned for its annual Carnival, which is one of the most spectacular and well-known events in the world. Paris is home to many iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Despite its size, Paris is also an important economic and political center of France, hosting many major French institutions and institutions of higher learning. The city has a rich cultural heritage and is known for its diverse food culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are a few potential trends that could shape the industry:
    
    1. Increased integration with human-like intelligence: As AI becomes more integrated with human-like intelligence, it could become more capable of mimicking human decision-making and emotions. This could lead to more nuanced and empathetic AI that can better understand and respond to complex human needs.
    
    2. Expansion of AI to include more natural language processing: With the rise of the internet and social media, AI is becoming increasingly dependent on natural language processing. This could lead to even more advanced AI that can better understand and respond to human language.
    
    3. Development of AI that can interact


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

     am

     a

     [

    Character

    's

     Name

    ]

     with

     a

     background

     in

     [

    Field

     or

     Area

     of

     Expert

    ise

    ].

     I

     have

     always

     been

     passionate

     about

     [

    Your

     Passion

    ]

     and

     strive

     to

     [

    Your

     Desired

     Outcome

    ].

     I

     am

     always

     on

     the

     lookout

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

     leader

    .

     What

    's

     your

     name

     and

     what

    's

     your

     background

    ?

     I

    'm

     excited

     to

     have

     the

     opportunity

     to

     meet

     you

    !

     [

    Your

     Name

    ]

     What

    's

     your

     name

     and

     what

    's

     your

     background

    ?

     [

    Your

     Name

    ]

     I

     am

     a

     [

    Your

     Name

    ],

     a

     [

    Character

    's

     Name

    ]

     with

     a

     background

     in

     [

    Field

     or

     Area

     of

     Expert

    ise

    ].

     I

     have

    
    
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

     Europe

     by

     population

     and

     is

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

     Paris

     is

     also

     a

     major

     center

     for

     international

     business

     and

     finance

    ,

     and

     has

     been

     home

     to

     numerous

     notable

     French

     people

    ,

     including

     monarch

    s

    ,

     presidents

    ,

     and

     political

     figures

    .

     It

     is

     home

     to

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     a

     UNESCO

     Creative

     City

    .

     Its

     diverse

     culture

     and

     impressive

     architecture

     make

     it

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     It

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

     for

     its

     romantic

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     looking

     bright

     with

     many

     advancements

     being

     made

    ,

     and

     there

     are

     several

     potential

     trends

     we

     can

     expect

     to

     see

    :
    


    1

    .

     Increased

     efficiency

     and

     productivity

    :

     AI

     is

     expected

     to

     continue

     improving

     efficiency

     and

     productivity

     across

     industries

    .

     We

     can

     expect

     to

     see

     more

     applications

     of

     AI

     in

     areas

     such

     as

     manufacturing

    ,

     healthcare

    ,

     and

     transportation

    ,

     which

     will

     lead

     to

     greater

     economic

     growth

     and

     job

     creation

    .
    


    2

    .

     Integration

     of

     AI

     into

     everyday

     life

    :

     AI

     is

     already

     being

     integrated

     into

     many

     aspects

     of

     our

     daily

     lives

    ,

     such

     as

     voice

     assistants

    ,

     self

    -driving

     cars

    ,

     and

     virtual

     assistants

    .

     We

     can

     expect

     to

     see

     even

     more

     integration

     in

     the

     coming

     years

    ,

     such

     as

     AI

    -powered

     personal

     assistants

    ,

     smart

     homes

    ,

    



```python
llm.shutdown()
```
