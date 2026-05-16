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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.73it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.72it/s]


    2026-05-16 18:26:51,672 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 18:26:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 23.87it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 32.91it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 32.91it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 32.91it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 32.91it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 32.91it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 32.91it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 32.91it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 32.91it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.91it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.99it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=73.02 GB):  21%|██        | 12/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.01 GB):  21%|██        | 12/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.01 GB):  21%|██        | 12/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.01 GB):  21%|██        | 12/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.00 GB):  21%|██        | 12/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.00 GB):  21%|██        | 12/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.08it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.14it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.14it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.14it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.14it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.14it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.14it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.89it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.89it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=320 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.44it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.44it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.44it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.44it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.44it/s]Capturing num tokens (num_tokens=176 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.44it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.51 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=112 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.77it/s] Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  81%|████████  | 47/58 [00:01<00:00, 46.27it/s]

    Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.49it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.49it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.49it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.49it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.49it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.49it/s] Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=4 avail_mem=72.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=4 avail_mem=72.46 GB): 100%|██████████| 58/58 [00:01<00:00, 40.63it/s]


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
    Generated text:  Carl and I have been studying for a month now and I was wondering if you could check my answer to this question? A glass of water that is 10 degrees Celsius will turn into steam at a temperature of 100 degrees Celsius. What is the entropy change for this process? A. -75 J/K B. 0 J/K C. 112 J/K D. -112 J/K
    
    To determine the entropy change for the process of water turning from 10 degrees Celsius to 100 degrees Celsius, we need to consider the process as a heat transfer process, which can
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the highest, executive branch of the government. Which of the following best describes the powers of the president?
    A) The power of the president is limited to the enforcement of the laws.
    B) The president has the authority to do what they want with the laws.
    C) The president has the authority to enforce laws, but they cannot make decisions about certain types of cases.
    D) The president has the authority to do what they want with the laws and make decisions about specific types of cases.
    
    To determine the best description of the powers of the president, let's analyze each option step by step:
    
    A) The power of
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. London
    C. Moscow
    D. New York
    Answer:
    
    A
    
    The first time there was a big discussion on whether 'freedom is absolute' in China was during the ____.
    A. New Culture Movement
    B. May Fourth Movement
    C. Cultural Revolution
    D. First Sino-American Trade War
    Answer:
    
    B
    
    The Chinese medicinal herb that is not accompanied by a fruit peel is ____.
    A. Fennel
    B. Ligusticum chuanxiong
    C. Sichuan pepper
    D. White wood鳖
    Answer:
    
    C
    
    Which of the
    ===============================
    Prompt: The future of AI is
    Generated text:  about interacting with others in new ways, and in 2019, this has been realized with the rise of AI-powered chatbots. On the one hand, we have seen numerous success stories of companies leveraging AI to improve the customer experience, increase efficiency, and improve the overall organizational effectiveness. On the other hand, we have also seen some challenges and concerns with AI.
    
    ### High-pressure Environments for AI
    
    There are a few high-pressure environments for AI where the problem comes in two distinct ways: the incompatibility of the data and the environment, and the data with the environment.
    
    ### Data and the Environment
    
    High-pressure


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [job title] at [company name], and I have been working in the [industry] for [number of years] years. I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in the industry. What do you do for a living? I'm a [job title] at [company name], and I have been working in the [industry] for [number of years] years.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art scene, and food culture. Paris is a vibrant and diverse city with a population of over 1. 5 million people. It is a major transportation hub and a major tourist destination. The city is home to many world-renowned museums, art galleries, and theaters. Paris is a city that is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations. This could lead to more efficient and effective decision-making, as well as better human-computer interaction.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability. This will require developers to be
    


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
    Generated text:  [insert character's name]. I am an experienced consultant with over ten years of experience in marketing and sales. My marketing and sales skills have been honed through countless hours of training and experience. I am passionate about helping businesses grow and achieving their goals, and I am committed to providing exceptional service to all clients. In my spare time, I enjoy hiking, painting, and spending time with my family. I am excited to meet you and learn more about your business! 🌟✨
    Hey there! 👋 I'm [insert character's name], a seasoned marketing and sales professional with over ten years of experience in the field.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    1. **Paris** is the capital city of France, located on the River Seine in the center of the country. It is the largest city and the most populous urban area in the European Union.
    
    2. **Historically**, Paris was the seat of the Holy Roman Empire, the Middle Ages capital of France, the seat of the French government, and the capital of France for several centuries, including the reign of Louis XIV.
    
    3. **Today**, Paris is a diverse, cultural, and historical city with a population of over 2.3 million people, making it the **most populous urban area in Europe**
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising, with various trends and developments expected to shape its trajectory. Here are some of the most notable trends in AI:
    
    1. Increased accuracy and efficiency: As AI systems become more sophisticated, they are becoming more accurate and efficient at performing tasks. This will lead to a more streamlined and streamlined business environment.
    
    2. Personalization and context-awareness: AI will increasingly be able to learn and adapt to the individual needs and preferences of users. This will allow for more personalized experiences and interactions.
    
    3. Automation of mundane tasks: AI will continue to automate tasks that are repetitive and time-consuming, freeing up workers to focus on more complex and creative


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

    'm

     a

     [

    Your

     occupation

    ]

     who

     has

     always

     been

     passionate

     about

     [

    Your

     favorite

     hobby

     or

     activity

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     What

     can

     you

     tell

     me

     about

     yourself

     and

     your

     interests

    ?


    [

    Your

     Name

    ]

     


    Your

     occupation

    :

     


    Favorite

     hobby

     or

     activity

    :

     


    What

     makes

     you

     tick

    ?


    I

    'm

     an

     AI

     language

     model

    ,

     programmed

     to

     assist

     users

     in

     generating

     text

     and

     answering

     questions

    .

     I

     don

    't

     have

     any

     personal

     experiences

     or

     interests

    .

     However

    ,

     I

    'm

     always

     ready

     to

     help

     and

     provide

     information

     to

     those

     who

     want

     to

     learn

    .

     Please

     feel

     free

     to

     ask

     me

     anything

     you

    'd

     like

     to

     know

    !

     #

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     commonly

     known

     as

     "

    The

     City

     of

     Light

    ."


    The

     statement

     succinct

    ly

     captures

     the

     essence

     of

     Paris

     as

     the

     cultural

     and

     economic

     hub

     of

     France

    ,

     known

     for

     its

     rich

     history

    ,

     vibrant

     art

     scene

    ,

     and

     status

     as

     a

     global

     met

    ropolis

    .

     The

     capital

    ,

     situated

     in

     the

     heart

     of

     the

     French

     Alps

    ,

     is

     a

     UNESCO

     World

     Heritage

     site

    ,

     and

     its

     influence

     can

     be

     seen

     in

     its

     influence

     on

     both

     French

     and

     international

     politics

    .

     France

    's

     capital

     is

     also

     home

     to

     several

     renowned

     universities

    ,

     including

     the

     University

     of

     Paris

    -S

    or

    bon

    ne

    ,

     which

     is

     one

     of

     the

     oldest

     and

     largest

     universities

     in

     the

     world

    .

     While

     the

     city

    's

     impact

     on

     the

     French

     economy

     is

     significant

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     a

     rapidly

     evolving

     landscape

     with

     numerous

     trends

     and

     developments

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     precision

     and

     accuracy

    :

     AI

     is

     becoming

     more

     precise

     and

     accurate

    ,

     with

     improvements

     in

     machine

     learning

     algorithms

     that

     enable

     AI

     systems

     to

     better

     understand

     and

     reason

     about

     complex

     data

    .
    


    2

    .

     Integration

     with

     human

     beings

    :

     AI

     is

     becoming

     more

     integrated

     with

     human

     beings

    ,

     enabling

     more

     complex

     and

     nuanced

     interactions

     between

     the

     two

    .
    


    3

    .

     Enhanced

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     ensuring

     privacy

     and

     security

     in

     the

     use

     of

     these

     systems

    .
    


    4

    .

     Autonomous

     systems

    :

     Autonomous

     systems

    ,

     which

     are

     AI

     systems

     that

    



```python
llm.shutdown()
```
