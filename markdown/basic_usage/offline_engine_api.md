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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.85it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.42it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.92it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.08it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.08it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.08it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.08it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.08it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.08it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.08it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.08it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.08it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.08it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 24.08it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.55it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.42 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.42 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.18 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.18 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.18 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.18 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.39it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.18 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.39it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.17 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.17 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.39it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=59.17 GB):  21%|██        | 12/58 [00:00<00:01, 30.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.17 GB):  21%|██        | 12/58 [00:00<00:01, 30.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.17 GB):  21%|██        | 12/58 [00:00<00:01, 30.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.16 GB):  21%|██        | 12/58 [00:00<00:01, 30.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.16 GB):  21%|██        | 12/58 [00:00<00:01, 30.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.15 GB):  21%|██        | 12/58 [00:00<00:01, 30.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.15 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.15 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.15 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.15 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.13 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=960 avail_mem=59.14 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.00it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=59.14 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.79it/s]Capturing num tokens (num_tokens=896 avail_mem=59.14 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.79it/s]Capturing num tokens (num_tokens=832 avail_mem=59.14 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.79it/s]Capturing num tokens (num_tokens=768 avail_mem=59.13 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.79it/s]Capturing num tokens (num_tokens=704 avail_mem=59.13 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.79it/s]Capturing num tokens (num_tokens=640 avail_mem=59.13 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.79it/s]Capturing num tokens (num_tokens=640 avail_mem=59.13 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.57it/s]Capturing num tokens (num_tokens=576 avail_mem=59.13 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.57it/s]Capturing num tokens (num_tokens=512 avail_mem=59.11 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.57it/s]Capturing num tokens (num_tokens=480 avail_mem=59.13 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.57it/s]Capturing num tokens (num_tokens=448 avail_mem=59.12 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.57it/s]

    Capturing num tokens (num_tokens=416 avail_mem=59.12 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.57it/s]Capturing num tokens (num_tokens=416 avail_mem=59.12 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=384 avail_mem=59.12 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=352 avail_mem=59.11 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=320 avail_mem=59.11 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=288 avail_mem=59.11 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=256 avail_mem=59.10 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=256 avail_mem=59.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.50it/s]Capturing num tokens (num_tokens=240 avail_mem=59.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.50it/s]

    Capturing num tokens (num_tokens=224 avail_mem=61.70 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.50it/s]Capturing num tokens (num_tokens=208 avail_mem=61.70 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.50it/s]Capturing num tokens (num_tokens=192 avail_mem=61.70 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.50it/s]Capturing num tokens (num_tokens=192 avail_mem=61.70 GB):  71%|███████   | 41/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=176 avail_mem=61.69 GB):  71%|███████   | 41/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=160 avail_mem=61.69 GB):  71%|███████   | 41/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=144 avail_mem=61.69 GB):  71%|███████   | 41/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=128 avail_mem=61.68 GB):  71%|███████   | 41/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=112 avail_mem=61.68 GB):  71%|███████   | 41/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=112 avail_mem=61.68 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=96 avail_mem=61.68 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.63it/s] Capturing num tokens (num_tokens=80 avail_mem=61.67 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.63it/s]

    Capturing num tokens (num_tokens=64 avail_mem=61.67 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=48 avail_mem=61.67 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=32 avail_mem=61.66 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=32 avail_mem=61.66 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.18it/s]Capturing num tokens (num_tokens=28 avail_mem=61.66 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.18it/s]Capturing num tokens (num_tokens=24 avail_mem=61.66 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.18it/s]Capturing num tokens (num_tokens=20 avail_mem=61.65 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.18it/s]Capturing num tokens (num_tokens=16 avail_mem=61.65 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.18it/s]Capturing num tokens (num_tokens=12 avail_mem=61.65 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.18it/s]Capturing num tokens (num_tokens=12 avail_mem=61.65 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.44it/s]Capturing num tokens (num_tokens=8 avail_mem=61.65 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.44it/s] Capturing num tokens (num_tokens=4 avail_mem=61.64 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.44it/s]

    Capturing num tokens (num_tokens=4 avail_mem=61.64 GB): 100%|██████████| 58/58 [00:01<00:00, 36.84it/s]


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
    Generated text:  Sam and I'm 15 years old. My parents are both doctors. Now, I know that doctors are very important, and the world needs doctors in need. So I'm really happy to study here. I have studied hard at school and I really want to be a doctor. I want to help people. I want to make sure everyone is healthy. I want to help sick people feel better. I want to be able to help people I love. I like to help sick people and I want to help sick people. What do you think is the best way to be a doctor? I think the best way to be a
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. There are 20 states in the country. Each state has a different number of senators. In 2020, each state had a different number of senators as well. By 2025, the number of senators per state had grown to 25. What is the total number of senators in all the states combined in 2025?
    
    To determine the total number of senators in all the states combined in 2025, we need to follow these steps:
    
    1. Identify the number of senators per state in 2020.
    2. Determine the number
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of ______.
    A. Bordeaux
    B. Paris
    C. St. Louis
    D. England
    Answer:
    
    B
    
    In a company's information system, the audit plan of the accounting department should primarily target the following areas: ( ) A. Ledger and accounting system B. Information systems in other departments C. Internal control over financial transactions D. Information technology
    A. In a company's information system, the audit plan of the accounting department should primarily target the following areas: ()
    B. Ledger and accounting system
    C. Information systems in other departments
    D. Internal control over financial transactions
    Answer:
    
    C
    
    The efficacy
    ===============================
    Prompt: The future of AI is
    Generated text:  being heavily driven by big data, as most of the industries and companies have started to utilize the technology in a more significant way. This technology can be used for a wide range of purposes, including healthcare, finance, and security. In the next few years, the use of AI will continue to grow, and it will become more diverse and sophisticated. This article will provide an overview of the future of AI and how it will impact various industries and sectors.
    The future of AI is highly dependent on the availability and integration of data. In the past, AI systems have been limited by the data they are working with. However, with the advent


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French Riviera. Paris is a cultural and economic center with a rich history dating back to the Roman Empire and the French Revolution. It is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its diverse cuisine, fashion, and art scene. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The French language is widely spoken, and the city is home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be increased emphasis on ethical and social considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as greater transparency and accountability in AI systems.
    
    3. Increased focus on AI ethics
    


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
    Generated text:  Sarah and I am a self-employed freelance graphic designer. I specialize in creating unique and visually appealing designs for clients all over the world. I have a passion for art and design and I believe that creativity is the key to success. I love to work outside the lines and take risks when it comes to getting things right. I have a knack for making my clients' projects look fantastic and I'm always looking for new opportunities to grow and learn. Thank you for asking to meet me, and I hope to meet you soon! **Sarah's Persona:** As a self-employed freelance graphic designer with a passion for art and design, Sarah is excited
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and the capital of France. It is known as "La Grande Enceinte" or "La Grande Mairie" due to its size and architecture. The city has a rich history, including the Eiffel Tower and many historical landmarks, including Notre-Dame Cathedral and the Palace of Versailles. Paris is a popular tourist destination and the seat of government, culture, and politics in France. It is also known for its art, music, and fashion. The city is home to millions of people and has a diverse population, with various ethnic groups and nationalities. Paris is often referred
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be a highly dynamic and diverse field. Here are some potential trends that may be experienced in the coming years:
    
    1. Increased focus on ethics and privacy: With the increasing amount of data being collected and analyzed, it is likely that AI will become more ethical and transparent. This will require greater emphasis on privacy and data protection, and will lead to stricter regulations and guidelines around AI use.
    
    2. Integration of AI with other technologies: AI is set to become more integrated with other technologies, such as autonomous vehicles, smart homes, and healthcare. This will require new approaches and tools to ensure that AI is used safely and effectively.
    
    3


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

    job

     title

     or

     profession

    ]

     at

     [

    company

     name

    ].

     I

     am

     excited

     to

     meet

     you

     and

     explore

     the

     possibilities

     that

     lie

     ahead

     as

     we

     embark

     on

     this

     journey

     together

    .

     Let

    's

     connect

    ,

     and

     let

    's

     create

     something

     beautiful

     together

    .

     [

    Name

    ]

     wants

     to

     learn

     more

     about

     [

    company

     name

    ],

     and

     I

     am

     happy

     to share

     my

     knowledge

     and

     experience

     in

     [

    job

     title

     or

     profession

    ].

     I

     am

     [

    age

    ]

     and

     I

     have

     [

    number

     of

     years

     of

     experience

     or

     education

    ].

     I

     am

     a

     [

    general

     attribute

    ],

     and

     I

     am

     always

     eager

     to

     learn

     and

     grow

    .

     Thank

     you

     for

     your

     time

    ,

     and

     I

     look

     forward

     to

     our

     future

     together

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    F

    acts

     about

     Paris

    :
    


    -

     Founded

     in

     

    7

    8

    9

     by

     Charles

     Mart

    el

    ,

     it

     is

     the

     third

     most

     populous

     city

     in

     the

     European

     Union

    .


    -

     It

     is

     the

     seat

     of

     the

     French

     government

    ,

     the

     French

     government

    's

     residence

    ,

     and

     the

     seat

     of

     the

     French

     Supreme

     Court

    .


    -

     It

     is

     the

     seat

     of

     the

     French

     parliament

    ,

     the

     Chamber

     of

     De

    puties

    ,

     and

     the

     headquarters

     of

     the

     French

     administrative

     hierarchy

    .


    -

     Paris

     is

     the

     most

     important

     city

     in

     France

     and

     the

     world

    .

     It

     is

     also

     the

     second

     most

     visited

     city

     in

     the

     world

    .


    -

     It

     has

     a

     rich

     cultural

     and

     historical

     heritage

    ,

     known

     for

     its

     art

    ,

     architecture

    ,

     and

     cuisine

    .

     


    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     significant

     advancements

     in

     several

     key

     areas

    ,

     driven

     by

     a

     combination

     of

     technological

     breakthrough

    s

    ,

     shifts

     in

     the

     business

     landscape

    ,

     and

     changing

     societal

     attitudes

     towards

     AI

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

     that

     we

     can

     expect

     to

     see

    :
    


    1

    .

     Increased

     Use

     of

     AI

     for

     Personal

    ized

     Medicine

    :

     AI

     can

     be

     used

     to

     analyze

     large

     datasets

     and

     provide

     personalized

     treatment

     recommendations

     for

     patients

    .

     For

     example

    ,

     AI

     can

     analyze

     medical

     images

    ,

     genetic

     information

    ,

     and

     other

     data

     to

     identify

     specific

     genetic

     or

     environmental

     factors

     that

     influence

     a

     patient

    's

     disease

     risk

    .

     This

     could

     lead

     to

     more

     effective

     personalized

     medicine

    ,

     better

     patient

     outcomes

    ,

     and

     fewer

     medical

     errors

    .
    


    2

    .

     More

     Natural

    



```python
llm.shutdown()
```
