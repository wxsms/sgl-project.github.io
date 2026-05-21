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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.95it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.24it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.24it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.24it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.24it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.24it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.24it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.24it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.24it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.24it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 23.24it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 23.24it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 32.41it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 32.41it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 32.41it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 32.41it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 32.41it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 32.41it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 32.41it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 32.41it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 32.41it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 32.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.91it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.91it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.91it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.91it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.91it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 36.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 36.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 36.32it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.32it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.32it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.38it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.38it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.38it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.38it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.38it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.38it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.45it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.31it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=144 avail_mem=72.44 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.97it/s]Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.90it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.90it/s]

    Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.90it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.90it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.90it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.90it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.28it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 39.36it/s]


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
    Generated text:  Xiaohong. I'm 13 years old and I'm from Fujian, China. I like reading and watching movies. I also like going to the beach. My favorite season is summer, and I like to stay at home with my family. My parents work at a big company in the city. My parents like to work very long hours and it's very tiring for them. I love the change from the work. My parents always go on vacations, but they always take me. I love to spend time with them and help them. I think their work is very important. The best way to spend my time is
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to finalize a deal with an international organization to reduce global greenhouse emissions. The organization, known as the Paris Agreement, has a set of emissions reduction goals for each country. If the United States aims to reduce its greenhouse emissions by 20% by 2025, how much would the United States need to reduce its emissions to meet this goal? Furthermore, if the Paris Agreement requires a global average temperature rise of 2 degrees Celsius by 2050 to ensure global sustainability, how many degrees Celsius in absolute terms would the United States need to reduce its emissions to meet both the country's goal and the global temperature
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is the city of the Eiffel Tower. Its French name is Paris, which has French origin. The capital of South Africa is Pretoria, which has English origin. A few other languages that can be heard are Chinese, Persian, English, Arabic, and German. Is this statement correct?
    
    To determine if the statement "The capital of South Africa is Pretoria, which has English origin" is correct, let's break down the information provided and analyze it step-by-step.
    
    1. **Identify the key elements of the statement:**
       - The capital of South Africa: Pretoria
       - The language
    ===============================
    Prompt: The future of AI is
    Generated text:  not in the computer - it's in the person. AI cannot exist in a vacuum, but rather in a community of people. And it's clear that AI research and development is a community activity, and that this community of people is determined to be more cooperative than competitive.
    
    According to the passage, what do people believe is the future of AI?
    
    A. It will continue to grow in the computer
    B. It will remain as a competition
    C. It will be a community activity
    D. It will be in a community of people
    
    Choose the word that best fits the blank in the above passage.  Answer according to:


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I'm always looking for ways to [job title] new things. I'm always eager to learn and grow, and I'm always looking for opportunities to contribute to the team. What's your favorite hobby or activity? I'm always looking for new things to try, and I'm always eager to learn and grow. What's your favorite book or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling city with a diverse population and is a major tourist destination. It is home to many famous French artists, writers, and musicians. The city is also known for its cuisine, including French cuisine, and is a popular destination for food lovers. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has been a hub of culture and politics
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for workers and businesses.
    
    2. Enhanced privacy and security: As AI technology becomes more advanced, we can expect to see increased emphasis on privacy and security. This could lead to new regulations and standards for AI development and use, as well as
    


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
    Generated text:  [insert name]. I'm a [insert job title or profession] who has been working for [insert company name] for [insert number of years] years. Currently, I'm [insert current job title or position] and have been with the company for [insert number of years] years. I'm passionate about [insert one or two specific hobbies, interests, or passions] and always strive to improve myself and the company I work for. I'm excited to learn more about [insert company's products, services, or industry] and make a difference in the world.
    
    What brings you to this position at [insert company name]?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a bustling metropolis with a rich cultural history dating back to the Middle Ages, and it continues to be a major international city today. It's also one of the world's most popular tourist destinations and a major economic center, with important institutions such as the French Academy and the French Embassy in the United States. 
    
    This statement provides a brief overview of the key features and historical significance of Paris as the capital city of France. The statement is concise, informative, and provides relevant information about a key
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities and potential applications. Here are a few potential trends that are likely to come in the coming years:
    
    1. Advancements in AI ethics and legal standards: With the increased focus on AI in legal and regulatory contexts, we can expect to see more rigorous scrutiny and regulation of AI development and deployment. This could lead to more stringent ethical guidelines and legal standards that AI systems need to meet in order to be considered safe and legal.
    
    2. Increased use of AI in healthcare: As AI technology becomes more advanced and integrated into healthcare, we can expect to see more AI-driven solutions for diseases and conditions that are currently difficult or impossible


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

    insert

     age

    ,

     gender

    ,

     occupation

    ,

     etc

    .

    ].

     I

     have

     been

     passionate

     about

     [

    insert

     interest

     or

     hobby

    ].

     I

     am

     passionate

     about

     [

    insert

     hobby

    ].

     I

     am

     [

    insert

     role

     in

     the

     company

    /

    organization

    ].

     I

     am

     [

    insert

     personal

     value

     or

     action

    ].

     I

     am

     [

    insert

     industry

    ],

     [

    insert

     position

    ].

     I

     am

     [

    insert

     resume

     link

    ].

     I

     am

     [

    insert

     personal

     value

     or

     action

    ].

     This

     is

     my

     [

    insert

     title

     or

     honorary

     title

    ],

     [

    insert

     summary

     of

     skills

    ,

     experience

    ,

     etc

    .

    ].

     I

     am

     [

    insert

     personal

     value

     or

     action

    ].

     I

     am

     [

    insert

     personal

     value

     or

     action

    ].

     This

     is

     my

     [

    insert

     title

     or

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     statement

     is

     concise

    ,

     precise

    ,

     and

     addresses

     the

     core

     topic

     of

     the

     question

     in

     a

     single

    ,

     clear

     sentence

    .

     Additionally

    ,

     it

     mentions

     that

     Paris

     is

     the

     capital

     of

     France

     and

     is

     a

     major

     city

     in

     the

     country

    ,

     as

     well

     as

     giving

     a

     specific

     name

     for

     it

    .

     This

     statement

     is

     appropriate

     for

     use

     in

     a

     formal

     or

     academic

     context

    .

     
    


    Sentence

    :

     "

    The

     capital

     of

     France

     is

     Paris

    ."

     
    


    This

     answer

     meets

     all

     the

     criteria

     for

     a

     concise

     and

     factual

     statement

     about

     Paris

    's

     capital

     city

    .

     It

     provides

     a

     clear

     and

     specific

     answer

     to

     the

     question

    ,

     uses

     the

     correct

     name

     for

     Paris

    ,

     and

     indicates

     that

     Paris

     is

     the

     capital

     of

     France

    .

     The

     statement

    
    
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

     including

    :
    


     

     

    1

    .

     Adv

    ancements

     in

     machine

     learning

     and

     artificial

     intelligence

    :

     As

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     continued

     improvements

     in

     machine

     learning

     algorithms

     and

     AI

     systems

    ,

     leading

     to

     even

     more

     sophisticated

     and

     accurate

     capabilities

    .


     

     

    2

    .

     Integration

     of

     AI

     into

     various

     industries

    :

     AI

     is

     already

     being

     used

     in

     a

     wide

     range

     of

     industries

    ,

     from

     healthcare

     to

     finance

     to

     manufacturing

    .

     We

     can

     expect

     to

     see

     continued

     integration

     of

     AI

     into

     more

     industries

     in

     the

     future

    ,

     leading

     to

     even

     more

     widespread

     use

     of

     AI

    -powered

     technologies

    .


     

     

    3

    .

     Enhanced

     safety

     and

     security

     of

     AI

     systems

    :

     As

     AI

    



```python
llm.shutdown()
```
