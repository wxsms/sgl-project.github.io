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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.76it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.75it/s]


    2026-05-20 07:45:25,124 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 07:45:25] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.56it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.56it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.56it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.56it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.56it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.56it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.56it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.56it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.56it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 13.79it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 13.79it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 13.79it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 13.79it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 13.79it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 13.79it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 13.79it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 13.79it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 19.99it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 19.99it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 19.99it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 19.99it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 19.99it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 19.99it/s]

    Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 19.99it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 19.99it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 19.99it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 27.12it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 27.12it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 27.12it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 27.12it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 27.12it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 27.12it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 27.12it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 27.12it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 27.12it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 34.61it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 34.61it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 34.61it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 34.61it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 34.61it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 34.61it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 34.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=66.57 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=66.54 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=66.54 GB):   3%|▎         | 2/58 [00:00<00:04, 13.90it/s]Capturing num tokens (num_tokens=7168 avail_mem=66.53 GB):   3%|▎         | 2/58 [00:00<00:04, 13.90it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=66.53 GB):   3%|▎         | 2/58 [00:00<00:04, 13.90it/s]Capturing num tokens (num_tokens=6656 avail_mem=66.53 GB):   7%|▋         | 4/58 [00:00<00:04, 11.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=66.53 GB):   7%|▋         | 4/58 [00:00<00:04, 11.21it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=66.53 GB):   7%|▋         | 4/58 [00:00<00:04, 11.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=66.53 GB):  10%|█         | 6/58 [00:00<00:05,  9.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.27 GB):  10%|█         | 6/58 [00:00<00:05,  9.67it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.26 GB):  10%|█         | 6/58 [00:00<00:05,  9.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.26 GB):  10%|█         | 6/58 [00:00<00:05,  9.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.26 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.57it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.26 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.25 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.57it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.25 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.25 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.57it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.25 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.25 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.24 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.24 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.24 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.23 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.23 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.23 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.23 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.90it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.23 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.77it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.21 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.77it/s]Capturing num tokens (num_tokens=960 avail_mem=72.22 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.77it/s] Capturing num tokens (num_tokens=896 avail_mem=72.22 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.77it/s]Capturing num tokens (num_tokens=832 avail_mem=72.21 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.77it/s]Capturing num tokens (num_tokens=768 avail_mem=72.21 GB):  34%|███▍      | 20/58 [00:01<00:01, 24.77it/s]Capturing num tokens (num_tokens=768 avail_mem=72.21 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.83it/s]Capturing num tokens (num_tokens=704 avail_mem=72.21 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.83it/s]Capturing num tokens (num_tokens=640 avail_mem=72.20 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.83it/s]Capturing num tokens (num_tokens=576 avail_mem=72.20 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.83it/s]Capturing num tokens (num_tokens=512 avail_mem=72.19 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.83it/s]

    Capturing num tokens (num_tokens=480 avail_mem=72.20 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.83it/s]Capturing num tokens (num_tokens=480 avail_mem=72.20 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=448 avail_mem=72.20 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=416 avail_mem=72.20 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=384 avail_mem=72.20 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=352 avail_mem=72.19 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=320 avail_mem=72.19 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=320 avail_mem=72.19 GB):  60%|██████    | 35/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=288 avail_mem=72.19 GB):  60%|██████    | 35/58 [00:01<00:00, 36.85it/s]

    Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.96it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.96it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.96it/s]

    Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.96it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.96it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.96it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.02it/s]

    Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.02it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 29.05it/s]


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
    Generated text:  Amina and I am a Software Engineering student at Siena College. I am trying to solve a problem in C programming. The problem is to find the maximum value of x such that y = x^2 + 4x is a perfect square.
    
    I am having trouble solving this problem. It is quite a complex problem, and I am not sure how to approach it. 
    
    I have been using the expression x = sqrt(4x - y) to find the maximum value of x. However, I am not sure how to approach this problem. Can someone please help me with this? 
    
    Thank you for your time and help
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 45 years old. If the average lifespan of a person is currently 75 years, what will be the difference in age between the president and the average lifespan of the person in 5 years?
    
    To determine the difference in age between the president of the United States (45 years old) and the average lifespan of the person in 5 years, we can follow these steps:
    
    1. Calculate the president's age in 5 years.
    2. Subtract the president's age from the average lifespan in 5 years.
    
    First, we calculate the president's age in 5 years:
    \[ 45 \text{
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Rome
    C. London
    D. Berlin
    Answer:
    
    A
    
    According to the 'People's Republic of China Law on Safety in Production', for production and operation units that fail to set up obvious safety warning signs at production and operation sites and on relevant facilities and equipment where there are significant hazardous factors, the production and operation unit shall be ordered to make corrections within a time limit and may be fined ____.
    A. Between RMB 50,000 and RMB 100,000
    B. Between RMB 50,000 and R
    ===============================
    Prompt: The future of AI is
    Generated text:  bright but the first steps are already being taken. Some of the biggest players in the field, Google, Microsoft, IBM, and Amazon, are ramping up their focus on AI and experimenting with new ways to bring AI into everyday applications. Not only are they developing the technology, but they are also taking the risk to experiment with new ways to bring the technology into everyday life.
    The potential to revolutionize the way we live and work is at the forefront of the minds of these tech giants, and they are taking steps to make that happen. This is exciting news for everyone. The more we embrace and learn about AI, the more we


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has [Number of Years] years of experience in [Field]. I'm passionate about [What I Love About My Profession]. I'm [What I Do Best]. I'm a [What I'm Known For]. I'm [What I'm Proud of]. I'm [What I'm Looking Forward To]. I'm [What I'm Looking For]. I'm [What I'm Looking For]. I'm [What I'm Looking For]. I'm [What I'm Looking For]. I'm [What I'm Looking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is home to many famous French artists, writers, and musicians, and is a major hub for the French language and culture. Paris is also known for its fashion industry, with many famous designers and boutiques. The city is a major transportation hub, with many major highways and rail lines connecting
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased use of AI in healthcare: AI is already being used to improve patient care, from personalized treatment plans to automated diagnostic tools. As AI technology continues to improve, we can expect to see even more sophisticated applications in healthcare.
    
    2. Integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI technology continues to evolve, we can expect to see even more seamless integration into our daily routines.
    
    3.
    


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
    Generated text:  [Name] and I am [Age], a [Type of Job or Occupation] in [Industry or Field]. I am currently living in [City/State] and have always been an [occupation or hobby]. I am passionate about [reason for love/hobby/interest], and I hope to make [lasting impact or contribution] in [field or industry]. I enjoy [interest or hobby], and I believe that [why I love this field or industry]. I believe that [why I love this field or industry] is essential for [reason for love/hobby/interest], and I hope to be able to make [lasting impact
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral.
    Paris is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be very bright and wide-ranging, with many potential trends shaping its development and impact in the coming decades. Some of the possible future trends include:
    
    1. Increased automation and artificial general intelligence: As AI becomes more capable of performing tasks that were once performed by humans, it is likely that we will see an increase in automation and artificial general intelligence. This could lead to the development of completely new ways of interacting with technology and the world around us.
    
    2. More integration with the physical world: With the increasing use of AI in industries such as healthcare and finance, there is a potential for AI to be more integrated with the physical world


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

     friendly

     and

     helpful

     AI

     assistant

    .

     How

     can

     I

     assist

     you

     today

    ?

     You

     can

     ask

     me

     anything

    ,

     and

     I

    'll

     do

     my

     best

     to

     provide

     you

     with

     accurate

     and

     helpful

     information

    .

     Please

     let

     me

     know

     if

     you

     have

     any

     questions

     or

     concerns

    .

     Let

     me

     know

    !

     Hey

     there

    !

     It

    's

     great

     to

     meet

     you

    !

     My

     name

     is

     [

    Your

     Name

    ],

     and

     I

    'm

     here

     to

     help

     you

    .

     How

     can

     I

     assist

     you

     today

    ?

     You

     can

     ask

     me

     anything

     and

     I

    'll

     do

     my

     best

     to

     provide

     you

     with

     accurate

     and

     helpful

     information

    .

     If

     you

     have

     any

     questions

     or

     concerns

    ,

     please

     let

     me

     know

    .

     Let

     me

     know

    !

     Hey

    
    
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

     It

     is

     a

     historic

     city

     located

     on

     the

     left

     bank

     of

     the

     Se

    ine

     River

    ,

     near

     the

     River

     Plate

    .

     The

     city

     is

     known

     for

     its

     rich

     cultural

     history

    ,

     fine

     arts

    ,

     food

    ,

     and

     fashion

    .

     Paris

     is

     the

     third

    -largest

     city

     in

     France

     and

     the

     fourth

    -largest

     in

     the

     world

     by

     population

    .

     It

     is

     also

     the

     birth

    place

     of

     numerous

     famous

     figures

     such

     as

     the

     French

     President

    ,

     King

     Louis

     XV

    ,

     and

     Napoleon

     Bon

    ap

    arte

    .

     Paris

     is

     considered

     to

     be

     one

     of

     the

     most

     beautiful

     cities

     in

     the

     world

     and

     is

     home

     to

     many

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

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     dynamic

     and

     diverse

    ,

     driven

     by

     advances

     in

     technology

    ,

     data

    ,

     and

     computational

     power

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

     that

     are

     likely

     to

     shape

     the

     field

     in

     the

     coming

     years

    :
    


    1

    .

     Improved

     Natural

     Language

     Processing

     (

    N

    LP

    ):

     AI

     will

     continue

     to

     advance

     in

     N

    LP

    ,

     which

     will

     enable

     machines

     to

     understand

     and

     generate

     human

     language

     in

     new

     and

     innovative

     ways

    .

     This

     will

     be

     crucial

     for

     applications

     like

     chat

    bots

    ,

     virtual

     assistants

    ,

     and

     language

     translation

    .
    


    2

    .

     Autonomous

     systems

    :

     AI

     is

     already

     being

     used

     in

     self

    -driving

     cars

    ,

     robots

    ,

     and

     drones

    ,

     and

     it

     is

     expected

     that

     AI

     will

     become

     more

     widespread

     and

     intelligent

    .

     Autonomous

     systems

     will

    



```python
llm.shutdown()
```
