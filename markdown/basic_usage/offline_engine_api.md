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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.19it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.25it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.21it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.21it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.21it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 31.95it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 31.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.05 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.05 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.04 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.04 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.04 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.04 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.03 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=59.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.26it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.11it/s]

    Capturing num tokens (num_tokens=960 avail_mem=58.98 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.11it/s] Capturing num tokens (num_tokens=960 avail_mem=58.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.86it/s]Capturing num tokens (num_tokens=896 avail_mem=58.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.86it/s]Capturing num tokens (num_tokens=832 avail_mem=58.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.86it/s]Capturing num tokens (num_tokens=768 avail_mem=58.97 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.86it/s]Capturing num tokens (num_tokens=704 avail_mem=58.97 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.86it/s]Capturing num tokens (num_tokens=640 avail_mem=58.97 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.86it/s]Capturing num tokens (num_tokens=640 avail_mem=58.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.26it/s]Capturing num tokens (num_tokens=576 avail_mem=58.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.26it/s]Capturing num tokens (num_tokens=512 avail_mem=58.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.26it/s]Capturing num tokens (num_tokens=480 avail_mem=58.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.26it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.26it/s]Capturing num tokens (num_tokens=416 avail_mem=58.96 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.26it/s]Capturing num tokens (num_tokens=416 avail_mem=58.96 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=384 avail_mem=58.96 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=352 avail_mem=58.96 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=320 avail_mem=58.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=288 avail_mem=58.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=256 avail_mem=58.95 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=256 avail_mem=58.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.84it/s]Capturing num tokens (num_tokens=240 avail_mem=58.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.84it/s]Capturing num tokens (num_tokens=224 avail_mem=58.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.84it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.84it/s]Capturing num tokens (num_tokens=192 avail_mem=58.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.84it/s]Capturing num tokens (num_tokens=176 avail_mem=58.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.84it/s]Capturing num tokens (num_tokens=176 avail_mem=58.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=160 avail_mem=58.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=144 avail_mem=58.92 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=128 avail_mem=58.92 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=112 avail_mem=58.92 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=96 avail_mem=58.92 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.52it/s] Capturing num tokens (num_tokens=96 avail_mem=58.92 GB):  81%|████████  | 47/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=80 avail_mem=58.91 GB):  81%|████████  | 47/58 [00:01<00:00, 42.42it/s]

    Capturing num tokens (num_tokens=64 avail_mem=58.91 GB):  81%|████████  | 47/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=48 avail_mem=58.91 GB):  81%|████████  | 47/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=32 avail_mem=58.90 GB):  81%|████████  | 47/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=28 avail_mem=58.90 GB):  81%|████████  | 47/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=28 avail_mem=58.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=24 avail_mem=58.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=20 avail_mem=58.89 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=16 avail_mem=58.89 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=12 avail_mem=58.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=8 avail_mem=58.84 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.31it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=58.84 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=4 avail_mem=58.82 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=4 avail_mem=58.82 GB): 100%|██████████| 58/58 [00:01<00:00, 38.08it/s]


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
    Generated text:  Tanya and I'm a 16-year-old college student. I would like to ask you a question. What is your favorite subject? A. My parents B. My family C. My teachers D. My friends
    Answer: C
    
    What is the Chinese meaning of "Wheat"?
    A. Corn
    B. Rice
    C. Wheat
    D. Potatoes
    Answer: C
    
    Which of the following statements is correct?
    A. In a metal, the greater the number of covalent bonds, the stronger the metallic bond.
    B. When there is a significant difference in the distribution of positive and negative charges
    ===============================
    Prompt: The president of the United States is
    Generated text:  a wealthy man. He lives in a very rich house, which is the biggest house in the world. This house is made of gold, and it is 1, 245 feet long and 365 feet wide. If the president lives in his house for 14 days, how much money will he have earned? To determine how much money the president will have earned, we need to follow these steps:
    
    1. Calculate the total area of the house.
    2. Determine the total number of days the president lives in his house.
    3. Calculate the total amount of money earned by multiplying the total area of the
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. London
    C. Rome
    D. Tokyo
    
    To determine the capital of France, let's follow these steps:
    
    1. Identify the capital of France.
    2. Verify the answer by checking the options provided.
    
    Step 1: Identify the capital of France.
    The capital of France is Paris.
    
    Step 2: Verify the answer by checking the options provided.
    The options given are:
    A. Paris
    B. London
    C. Rome
    D. Tokyo
    
    The correct capital of France is Paris.
    
    Therefore, the answer is \boxed{A}.
    ===============================
    Prompt: The future of AI is
    Generated text:  in the making, with the technology rapidly advancing, and a number of companies are looking to capitalize on this opportunity.
    The terms “Data Science” and “AI” are not often used in conjunction, but they can be used interchangeably, and data science is an important component of AI.
    Data science and AI are increasingly being used in business operations to improve efficiency, predict trends, and support decision-making.
    AI is an emerging technology that is capable of processing and analyzing large amounts of data to improve decision-making processes.
    In today’s world, businesses are looking for ways to stay ahead of the competition and gain a competitive advantage. AI is a


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


    Generated text:  [Name] and I'm a [occupation] who has been [number of years] in the industry. I'm passionate about [reason for interest in the industry]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [number of years] in the industry and I'm always eager to learn and improve. I'm a [number of years] in the industry and I'm always eager to learn and improve. I'm a [number of years] in the industry and I'm always eager to learn and improve. I'm a [number of years] in the industry and I'm always eager to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub, with a diverse population and a rich history dating back to the Roman Empire. The city is known for its cuisine, fashion, and art, and is a popular tourist destination. It is also home to the world's largest metro system, the Paris Métro, which runs from the Eiffel Tower to the Louvre. Paris is a city of contrasts, with its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and preferences.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be increased concerns about privacy and security. Governments and organizations will need to develop new technologies and policies to protect the privacy and security of AI systems.
    
    3. Greater reliance on AI for decision-making: AI systems will become more integrated with human decision-making processes, allowing them
    


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
    Generated text:  [Name], and I'm a [Field] specialist. I've been working in this field for [X years] and have a [set of skills] that make me unique.
    I'm passionate about [Name] because I love [reason why it's important to me]. And I'm always looking for ways to [future goal] because I want to [future goal] more. If you're looking for a career that aligns with my values and passions, I'd be thrilled to help you discover it. Start the conversation! (Link to job posting) [Name] - Short, neutral self-introduction. Based on the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the central region of the country. It is the largest city in Europe and the third most populous in the world. The city is renowned for its rich history, architecture, and cultural landmarks, including the Eiffel Tower and the Louvre Museum. Paris is also known for its vibrant nightlife, fashion, and food scene. It has been a major economic hub and cultural center since the 12th century, and remains one of the world's most iconic cities. According to the 2021 French Census, Paris is home to an estimated population of over 2. 5 million residents.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly diverse and can be influenced by a multitude of factors, including advances in computing power, advancements in data and machine learning techniques, and shifts in societal and cultural values. Here are some possible trends that AI is likely to experience in the coming years:
    
    1. Increased focus on ethical AI: With the rise of ethical concerns around AI, there will be a greater emphasis on the ethical considerations surrounding AI development and deployment. This will likely lead to more rigorous testing and validation processes to ensure that AI systems are not harmful or unethical.
    
    2. Advances in natural language processing: With the increasing availability of large amounts of text data, natural language processing (


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

     name

    ]

     and

     I

    'm

     a

     [

    insert

     occupation

     or

     profession

    ]

     who

     has

     been

     in

     the

     [

    insert

     occupation

    ]

     for

     [

    insert

     number

     of

     years

    ]

     years

    .

     I

    'm

     [

    insert

     age

    ]

     years

     old

     and

     [

    insert

     nationality

     or

     ethnicity

    ].

     I

    've

     always

     been

     passionate

     about

     [

    insert

     something

     you

     enjoy

     doing

    ],

     and

     I

    'm

     always

     looking

     to

     learn

     new

     things

    .

     I

    'm

     [

    insert

     something

     you

    're

     knowledgeable

     on

    ],

     and

     I

    'm

     eager

     to

     share

     my

     knowledge

     with

     others

    .

     How

     would

     you

     describe

     yourself

    ?

     Hello

    ,

     my

     name

     is

     [

    insert

     name

    ]

     and

     I

    'm

     a

     [

    insert

     occupation

     or

     profession

    ]

     who

     has

     been

     in

     the

     [

    insert

     occupation

    ]

     for

     [

    insert

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     most

     populous

     city

     in

     France

     and

     the

     largest

     metropolitan

     area

     in

     Europe

    .

     Paris

     is

     known

     for

     its

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     The

     city

     has

     a

     rich

     history

     and

     culture

    ,

     including

     the

     influence

     of

     various

     French

     styles

     and

     art

     movements

    .

     Paris

     has

     a

     diverse

     population

     and

     a

     strong

     sense

     of

     community

    ,

     with

     many

     French

     people

     living

     in

     the

     surrounding

     areas

    .

     It

     is

     also

     a

     major

     financial

     center

     and

     one

     of

     the

     world

    's

     most

     important

     cultural

     and

     artistic

     centers

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     Site

     and

     one

     of

     the

     world

    's

     most

    -

    visited

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     dominated

     by

     a

     wide

     range

     of

     technological

     innovations

    ,

     both

     in

     terms

     of

     hardware

     and

     software

    .

     Some

     of

     the

     key

     trends

     that

     are

     likely

     to

     shape

     the

     future

     of

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     natural

     language

     processing

    :

     Natural

     language

     processing

     is

     becoming

     increasingly

     important

     as

     AI

     systems

     are

     required

     to

     understand

     and

     respond

     to

     human

     language

    .

     This

     trend

     is

     likely

     to

     continue

     as

     more

     applications

     are

     required

     to

     leverage

     the

     power

     of

     AI

    .
    


    2

    .

     Rise

     of

     AI

     in

     healthcare

    :

     With

     the

     increasing

     availability

     of

     large

     amounts

     of

     medical

     data

    ,

     AI

     is

     likely

     to

     play

     a

     more

     significant

     role

     in

     healthcare

    .

     This

     includes

     applications

     such

     as

     personalized

     medicine

    ,

     disease

     prediction

    ,

     and

     drug

     discovery

    



```python
llm.shutdown()
```
