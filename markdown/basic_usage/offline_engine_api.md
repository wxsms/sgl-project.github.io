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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.21it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:20,  4.56s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.24it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.53it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.64it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.64it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.64it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.64it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.64it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.64it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.64it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.64it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.64it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.64it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.64it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.59it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.59it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.59it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.59it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.59it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.59it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.59it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.59it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.59it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.59it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.48it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.48it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.48it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.48it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.48it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.48it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.48it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.48it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.48it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.22 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.22 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.21 GB):   9%|▊         | 5/58 [00:00<00:02, 21.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.21 GB):   9%|▊         | 5/58 [00:00<00:02, 21.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.76it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.19 GB):   9%|▊         | 5/58 [00:00<00:02, 21.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.19 GB):   9%|▊         | 5/58 [00:00<00:02, 21.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.58it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.58it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.58it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.18 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.18 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.16 GB):  31%|███       | 18/58 [00:00<00:01, 36.24it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.16 GB):  31%|███       | 18/58 [00:00<00:01, 36.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.16 GB):  31%|███       | 18/58 [00:00<00:01, 36.24it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.14 GB):  31%|███       | 18/58 [00:00<00:01, 36.24it/s]Capturing num tokens (num_tokens=960 avail_mem=72.15 GB):  31%|███       | 18/58 [00:00<00:01, 36.24it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.15 GB):  31%|███       | 18/58 [00:00<00:01, 36.24it/s]Capturing num tokens (num_tokens=896 avail_mem=72.15 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.20it/s]Capturing num tokens (num_tokens=832 avail_mem=72.15 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.20it/s]Capturing num tokens (num_tokens=768 avail_mem=72.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.20it/s]Capturing num tokens (num_tokens=704 avail_mem=72.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.20it/s]Capturing num tokens (num_tokens=640 avail_mem=72.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.20it/s]Capturing num tokens (num_tokens=576 avail_mem=72.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.20it/s]Capturing num tokens (num_tokens=576 avail_mem=72.14 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.02it/s]Capturing num tokens (num_tokens=512 avail_mem=72.12 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.02it/s]Capturing num tokens (num_tokens=480 avail_mem=72.14 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.02it/s]Capturing num tokens (num_tokens=448 avail_mem=72.14 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.02it/s]Capturing num tokens (num_tokens=416 avail_mem=72.13 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.02it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.13 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.02it/s]Capturing num tokens (num_tokens=384 avail_mem=72.13 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=352 avail_mem=72.13 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=320 avail_mem=72.12 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=288 avail_mem=72.12 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=256 avail_mem=72.12 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=240 avail_mem=72.11 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=240 avail_mem=72.11 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=224 avail_mem=72.11 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=208 avail_mem=72.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=192 avail_mem=72.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=176 avail_mem=72.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.57it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=160 avail_mem=72.10 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=144 avail_mem=72.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=128 avail_mem=72.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=112 avail_mem=72.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=96 avail_mem=72.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.15it/s] Capturing num tokens (num_tokens=80 avail_mem=72.08 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=80 avail_mem=72.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=64 avail_mem=72.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.16it/s]

    Capturing num tokens (num_tokens=48 avail_mem=72.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=32 avail_mem=72.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=28 avail_mem=72.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=24 avail_mem=72.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.16it/s]Capturing num tokens (num_tokens=24 avail_mem=72.07 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=20 avail_mem=72.06 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=16 avail_mem=72.06 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=12 avail_mem=72.06 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=8 avail_mem=72.05 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.34it/s] Capturing num tokens (num_tokens=4 avail_mem=72.05 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.34it/s]

    Capturing num tokens (num_tokens=4 avail_mem=72.05 GB): 100%|██████████| 58/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=4 avail_mem=72.05 GB): 100%|██████████| 58/58 [00:01<00:00, 38.17it/s]


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
    Generated text:  Mayse, and I am a science student, hoping to become a researcher. I am a big fan of sci-fi fiction and I enjoy reading about space travel. My main question is, what is the most difficult thing to do in space travel? Please provide your answer in one sentence. 
    Mayse is looking forward to future of space travel and would like to know what challenges researchers face in the field.
    As a space researcher, the most difficult thing to do in space travel is the isolation and lack of communication with the ground, which can lead to delays in decision-making and progress. This is a significant challenge that researchers face as they
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is like a king or queen of a country. Before the president of the United States takes office (the president of the United States takes office at the beginning of the year), the two main questions he or she needs to ask each other are: 1. What do you want to do? 2. What are you going to do? In those questions, the president wants to know the other person's idea. This is the reason that the president's first job is to listen to the people. After that, the president tells the other person what he or she wants to do and asks what the other
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Rome
    C. London
    D. Madrid
    Answer: A
    
    The balance sheet of a certain company shows that the total current assets are 10 million yuan, the total current liabilities are 6 million yuan, and the total long-term liabilities are 4 million yuan. The company's quick ratio is ____.
    A. 0.5
    B. 1
    C. 1.5
    D. 2
    Answer: B
    
    The risk of money laundering is ____.
    A. Moral risk
    B. Legal risk
    C. Systematic risk
    D. Market risk
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of researchers. This article will discuss the current status of the field and explore how it is progressing. The field of AI is growing rapidly, with the goal of making machines that can understand and reason like humans. It is expected that this will lead to new advancements in fields such as healthcare, finance, and transportation.
    The field of AI is focused on the development of machine learning algorithms that can process and analyze large amounts of data. These algorithms are designed to learn from data and make predictions or decisions based on that data. The goal is to create machines that can be used to solve problems in a variety of fields, from medicine


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Name] from [Location]. I'm a [Number] year old, [Name] from [Location]. I'm a [Number] year old, [Name] from [Location]. I'm a [Number] year old, [Name] from [Location]. I'm a [Number] year old, [Name] from [Location]. I'm a [Number] year old, [Name] from [Location]. I'm a [Number] year old, [Name] from [Location].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third-largest city in the world by population. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is also a major financial center and a major tourist destination. The city is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also known for its cuisine, including French cuisine and international cuisine. Paris is a city that is constantly evolving and changing, with new developments and attractions being added to the city's list of attractions. The city is also home to many cultural institutions, including museums,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: As AI becomes more advanced, it is likely to automate more and more tasks, from manufacturing to customer service. This could lead to job losses in certain industries, but also create new opportunities for people to work in areas like data analysis and machine learning.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be increased concerns about privacy and security. This could lead to new regulations and standards being put in place to protect
    


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
    Generated text:  [Name], and I'm a [Occupation] and [Name]. I'm currently [Age], and I've always been passionate about [X] because [X]. I'm [X], and I'm really happy to be here. What brings you to this world today? Let me know! #self-introduction
    
    Hello, my name is [Name], and I'm a [Occupation] and [Name]. I'm currently [Age], and I've always been passionate about [X] because [X]. I'm [X], and I'm really happy to be here. What brings you to this world today?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the second-largest city in the country and the most populous, with a population of around 2.3 million people. Paris is known for its historical landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and Montmartre, and its vibrant French culture and cuisine. It is also a major transportation hub and a major tourist destination. Paris is a UNESCO World Heritage site and has a rich history dating back to the Roman Empire. The city has a strong focus on education and culture, with a number of prestigious universities and museums located within the city limits. Paris is also home to a number of international institutions
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by increased reliance on AI, greater use of AI in natural language processing and machine learning, and continued development of new AI technologies. AI is likely to be integrated into a wide range of industries and applications, including healthcare, transportation, finance, and entertainment. As AI becomes more sophisticated, it is likely to be used to improve human intelligence, speed up decision-making, and solve complex problems in areas such as cybersecurity and climate change. AI is also likely to be used for the advancement of human knowledge, such as in the development of artificial intelligence systems for research and development. However, it is important to note that AI is


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

    First

     Name

    ]

     and

     I

    'm

     a

     [

    Last

     Name

    ]

    !

     I

    've

     been

     working

     in

     the

     field

     of

     [

    Role

    ]

     for

     [

    Number

     of

     Years

    ]

     years

    .

     My

     past

     projects

     include

     [

    List

     of

     projects

     completed

    ].

     What

     brings

     you

     to

     this

     industry

     now

    ?

     What

     do

     you

     like

     to

     do

     in

     your

     free

     time

    ?


    I

     can

    't

     provide

     any

     specific

     information

    ,

     but

     I

     would

     suggest

     starting

     with

     "

    Hello

    ,

     my

     name

     is

     [

    First

     Name

    ]

     and

     I

    'm

     a

     [

    Last

     Name

    ]

    !"

     and

     leaving

     the

     rest

     up

     to

     creativity

    .

     Remember

     to

     tailor

     your

     introduction

     to

     what

     you

     want

     to

     show

     about

     yourself

     and

     what

     makes

     you

     unique

    .

     Good

     luck

     with

     your

     self

    -int

    roduction

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     encaps

    ulates

     the

     core

     fact

     about

     Paris

     as

     the

     national

     capital

     of

     France

    ,

     making

     it

     clear

     and

     concise

     for

     the

     purpose

     of

     a

     brief

     summary

     or

     quote

    .

     Note

     that

     it

     may

     not

     be

     as

     detailed

     as

     a

     full

    -f

    ledged

     historical

     statement

     about

     the

     city

    ,

     but

     it

     succinct

    ly

     con

    veys

     its

     importance

     in

     French

     politics

     and

     culture

    .

     
    


    For

     a

     more

     detailed

     or

     in

    -depth

     statement

    ,

     consider

     the

     following

    :
    


    The

     capital

     of

     France

    ,

     Paris

    ,

     is

     known

     for

     its

     rich

     history

    ,

     iconic

     architecture

    ,

     and

     vibrant

     culture

    .

     
    


    This

     expanded

     version

     would

     provide

     a

     fuller

     picture

    ,

     allowing

     for

     a

     more

     accurate

     and

     informative

     statement

    .

     However

    ,

     it

     would

     still

     remain

     concise

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     marked

     by

     rapid

     advancements

     in

     several

     key

     areas

    ,

     including

    :
    


    1

    .

     Increased

     integration

     with

     human

     AI

    :

     With

     the

     development

     of

     machine

     learning

     and

     deep

     learning

    ,

     AI

     is

     expected

     to

     become

     more

     integrated

     with

     human

     AI

    .

     This

     could

     lead

     to

     a

     more

     seamless

     and

     efficient

     use

     of

     AI

     in

     various

     industries

    .
    


    2

    .

     Enhanced

     intelligence

     and

     autonomy

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     expected

     to

     become

     more

     intelligent

     and

     autonomous

    ,

     with

     the

     ability

     to

     make

     decisions

     and

     take

     actions

     without

     human

     intervention

    .
    


    3

    .

     Personal

    ization

     and

     context

    -aware

    ness

    :

     AI

     is

     expected

     to

     become

     more

     personalized

     and

     context

    -aware

    ,

     with

     the

     ability

     to

     learn

     and

     adapt

     to

     the

     user

    's

     behavior

     and

    



```python
llm.shutdown()
```
