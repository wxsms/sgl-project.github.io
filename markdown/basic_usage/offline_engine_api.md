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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.77it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<06:20,  6.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<06:20,  6.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:06<06:20,  6.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:06<06:20,  6.67s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:06<06:20,  6.67s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:53,  1.02s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:53,  1.02s/it]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:06<00:53,  1.02s/it]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:06<00:53,  1.02s/it]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:06<00:53,  1.02s/it]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:06<00:53,  1.02s/it]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:06<00:53,  1.02s/it]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:06<00:53,  1.02s/it]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:15,  2.98it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:15,  2.98it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:06<00:15,  2.98it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:06<00:15,  2.98it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:06<00:15,  2.98it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:06<00:15,  2.98it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:06<00:15,  2.98it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:06<00:15,  2.98it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:06<00:15,  2.98it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:06<00:15,  2.98it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:06<00:05,  6.43it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:06<00:05,  6.43it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:06<00:05,  6.43it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:07<00:05,  6.43it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:07<00:05,  6.43it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:07<00:05,  6.43it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:07<00:05,  6.43it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:07<00:05,  6.43it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:07<00:05,  6.43it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:07<00:05,  6.43it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:02, 10.84it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:02, 10.84it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:02, 10.84it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:02, 10.84it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:02, 10.84it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:02, 10.84it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:02, 10.84it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:07<00:02, 10.84it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:07<00:02, 10.84it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:07<00:02, 10.84it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:07<00:02, 10.84it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:07<00:01, 17.07it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:07<00:01, 17.07it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:07<00:01, 17.07it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:07<00:01, 17.07it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:07<00:01, 17.07it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:07<00:01, 17.07it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:07<00:01, 17.07it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:07<00:01, 17.07it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:07<00:01, 17.07it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:07<00:01, 17.07it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:07<00:01, 17.07it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:07<00:00, 24.44it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:07<00:00, 24.44it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 24.44it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 24.44it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:07<00:00, 24.44it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:07<00:00, 24.44it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:07<00:00, 24.44it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:07<00:00, 24.44it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:07<00:00, 24.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.78it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.10it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.45it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.48it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.65it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.15it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.15it/s]

    Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.15it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.15it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.15it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 33.38it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 33.38it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.38it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.38it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.38it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.70it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.86it/s] Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 33.15it/s]

    Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.37it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.37it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.37it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.37it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.37it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.37it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.93it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.93it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.93it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 32.96it/s]


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
    Generated text:  Rashid and I am a computer scientist and information security specialist. I specialize in performing risk analysis and identifying potential security vulnerabilities in applications. I recently worked on a project that involved analyzing the vulnerability of a system to various types of attacks. One of the vulnerabilities I found was an SQL injection attack. 
    
    Can you please provide an explanation and solution for the SQL injection vulnerability? 
    
    Furthermore, please address the following scenario:
    A company has implemented a new system that uses a database. The company's database contains sensitive data such as customer information, payment details, and personal information. The company has a strict policy to only allow certain users to access the
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. How many different ways can this occur if no two consecutive terms in a presidency can be consecutive terms? To determine the number of ways the president can run for a second term in a presidency where no two consecutive terms can be consecutive terms, we can use a recursive approach. Let's denote the number of valid second terms by \(a_n\).
    
    First, consider the base cases:
    - \(a_0 = 0\): No president can be the second term.
    - \(a_1 = 1\): The only valid president is the first president.
    
    For \(n \geq 2\
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Paris
    B. London
    C. New York
    D. Tokyo
    Answer:
    
    A
    
    Which of the following is NOT a function of a city?
    A. Political and cultural center
    B. Economic center
    C. Service center
    D. Cultural center
    Answer:
    
    D
    
    The primary function of a city is to ____
    A. Serve as a political and cultural center
    B. Serve as a service center
    C. Serve as a business center
    D. Serve as a cultural center
    Answer:
    
    A
    
    Which of the following is not a function of a city?
    A. Political and cultural center
    
    ===============================
    Prompt: The future of AI is
    Generated text:  highly uncertain. It’s important to stay informed, so that you can better understand the latest developments and to stay ahead of the curve. Here are a few questions to help you stay informed on the future of AI and the progress made so far:
    
      1. What is the current state of AI?
      2. What is the current research focus in AI?
      3. What is the current state of AI ethics and its role in AI development?
      4. What is the current state of AI development?
      5. What is the current state of AI deployment?
    
    Keep in mind that these are just a few


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. I enjoy [insert a short, positive description of your hobbies or interests]. What do you like to do in your free time? I like to [insert a short, positive description of your hobbies or interests]. I'm always looking for new experiences and learning opportunities. What's your favorite hobby or activity? I love [insert a short, positive description of your favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history dating back to the Roman era. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its fashion industry, with many famous designers and fashion houses operating in the area. Paris is a cultural and artistic center, with many museums, theaters, and art galleries, and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Increased use
    


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
    Generated text:  [Name] and I am a [Occupation] [Description of Occupation]. I have always been passionate about [Purpose of the Occupation], and I am dedicated to [Roles or Stances]. I am always willing to learn and grow from my experiences and failures, always staying true to [My Personal Values]. I have a strong work ethic, and I take pride in doing what I do. I am always up for a challenge, and I am always ready to push myself to the limit. I enjoy [My Favorite Activity/Exercise/Activity], and I love to socialize with people of different backgrounds and beliefs. I am always looking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city of lights. It is known for its rich history, famous landmarks, and unique culture. Paris is also a hub for the arts, fashion, and cuisine, and is home to numerous museums, theaters, and art galleries. Paris is a bustling metropolis with a population of over 1 million people, and it is the most visited city in the world. The city is also renowned for its fashion-forward dress code and is a popular destination for tourists from all over the world. French cuisine is also a major part of Paris, with its traditional French dishes and regional specialties. Paris is a cultural and intellectual center, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, and there is no guarantee of its progression. However, there are several possible trends that are likely to shape the AI landscape in the years to come.
    
    1. Increased focus on ethical AI: As AI becomes more and more complex and powerful, there will be increased pressure to ensure that it is used ethically and for the benefit of society as a whole. This may include creating regulations and guidelines for AI development, ensuring that AI is not used to harm or misdirect human behavior, and promoting transparency and accountability in AI systems.
    
    2. Greater use of AI in healthcare: AI is already being used to improve healthcare outcomes, from


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

    name

    ]

     and

     I

    'm

     a

     [

    job

     title

     or

     occupation

    ]

     with

     [

    number

     of

     years

     of

     experience

    ].

     I

    've

     always

     been

     fascinated

     by

     the

     world

     and

     how

     it

     works

    ,

     but

     I

    've

     never

     had

     the

     opportunity

     to

     dive

     deep

     into

     it

     like

     this

    .

     My

     goal

     is

     to

     become

     a

     [

    field

     or

     career

    ]

     and

     I

    'm

     eager

     to

     explore

     all

     the

     possibilities

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    fill

     in

     with

     specific

     details

     about

     your

     background

    ,

     education

    ,

     work

     experience

    ,

     personal

     experiences

    ,

     etc

    .

    ].

     [

    insert

     your

     passion

    ,

     hobbies

    ,

     interests

    ,

     etc

    .

    ].

     I

     am

     looking

     forward

     to

     exploring

     and

     discovering

     the

     world

    .

     I

     hope

     you

     will

     be

     able

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     unpredictable

     and

     will

     likely

     involve

     a

     number

     of

     different

     trends

     and

     developments

    .

     Here

     are

     some

     potential

     trends

     that

     may

     come

     in

     the

     next

     few

     years

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

     continue

     to

     become

     more

     sophisticated

     and

     personalized

    ,

     making

     healthcare

     more

     efficient

     and

     effective

    .

     AI

    -powered

     diagnostic

     tools

    ,

     virtual

     assistants

    ,

     and

     even

     personalized

     medicine

     may

     become

     more

     common

    .
    


    2

    .

     AI

     in

     finance

    :

     AI

     will

     become

     more

     prevalent

     in

     finance

    ,

     from

     fraud

     detection

     and

     risk

     management

     to

     trading

     algorithms

     and

     personalized

     investment

     recommendations

    .

     AI

    -driven

     trading

     platforms

     and

     intelligent

     algorithms

     will

     help

     improve

     market

     intelligence

    ,

     optimize

     portfolios

    ,

     and

     reduce

     costs

    .
    


    3

    .

     AI

     in

     manufacturing

    :

     AI

     will

     continue

    



```python
llm.shutdown()
```
