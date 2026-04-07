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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.01it/s]


    2026-04-07 16:51:53,955 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 16:51:53] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.17it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.17it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.17it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.17it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.17it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.17it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:02<00:07,  6.17it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:02<00:07,  6.17it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.17it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.51it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.61it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.61it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.61it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.61it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.61it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.61it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.61it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.09it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.36it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.36it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.36it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.36it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.36it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.36it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.36it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.32it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.32it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.32it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.32it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.32it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.32it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.32it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.32it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:02, 19.01it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 19.01it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 19.01it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 19.01it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.02it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.02it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.02it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  31%|███       | 18/58 [00:00<00:01, 26.45it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 26.45it/s] Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  31%|███       | 18/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.97it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.97it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.97it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.97it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.97it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:01<00:01, 31.97it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.95it/s]

    Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.89it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.89it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.89it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.89it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.89it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.89it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.75it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.75it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.75it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.75it/s]

    Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.75it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.75it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.39it/s] Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 40.87it/s]

    Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  81%|████████  | 47/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  81%|████████  | 47/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  81%|████████  | 47/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.11it/s] Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.45it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.45it/s]

    Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 33.13it/s]


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
    Generated text:  Jack and I am 22 years old and I am a software engineer. I am fluent in both Python and JavaScript. What are some common challenges faced by software developers and how do you overcome them?
    As a software engineer, I often find myself facing challenges in both programming and problem-solving. One common challenge is the need to keep up with the latest trends and technologies in the industry. This can be particularly challenging when there are a lot of updates happening in a short amount of time.
    Another challenge is managing a team of developers with different backgrounds and skills. It can be difficult to ensure that everyone is on the same page and working towards
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ____.
    A. Presidential candidate
    B. Head of state
    C. Head of government
    D. President of the Senate
    Answer:
    
    B
    
    Which of the following statements about anti-tuberculosis drugs is incorrect?
    A. They all have anti-tuberculosis effects
    B. They all inhibit the biosynthesis of nucleic acids
    C. They have a broad spectrum of activity against various types of bacteria
    D. They have a wide range of effects on the respiratory tract
    E. They can be used alone or combined with others for treatment
    Answer:
    
    B
    
    In the process of using paper chromatography for separating pigments
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Paris, also known as "la Ruine des Orangers" (The Ruins of the Orangers), is one of the most important cities of the world and has been a major metropolis since ancient times. It was founded by the Gauls and ancient French aristocrats, and became a major metropolis during the Middle Ages. It is famous for its landmarks, including the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and Arc de Triomphe.
    
    Paris is the capital of France. The seat of government of France, the capital of the French Departments, the capital of the French provinces, and
    ===============================
    Prompt: The future of AI is
    Generated text:  full of wonder and excitement. Think about the potential that could exist with the discovery of advanced AI, the possibility of creating new technologies that don’t exist now, and how much more we might be able to learn and improve about ourselves. But as with all things new and exciting, there are a few things to consider before getting too excited about it.
    One of the biggest fears with the potential of artificial intelligence is that it could replace us completely. The job market could grind to a halt, and most jobs would be automated. This could cause widespread unemployment and lower quality of life for many people. While it’s difficult to say for certain what


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [industry]. I'm a [occupation] who is always looking for ways to [describe a new skill or hobby]. I'm a [occupation] who is always looking for ways to [describe a new skill or hobby]. I'm a [occupation] who is always looking for ways to [describe a new skill or hobby]. I'm a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville-Marie" or "La Ville de Paris". It is the largest city in France and the second-largest city in the European Union. Paris is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among other landmarks. The city is known for its rich history, art, and culture, and is a major tourist destination. It is also home to many important political and economic institutions, including the French Parliament and the French National Assembly. Paris is a vibrant and dynamic city, with a diverse population and a rich cultural heritage. It is a major hub for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the development of AI in the coming years:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability. AI developers will need to be more mindful of the potential impact of their technology on society and work to ensure that it is used in a way that is fair and just.
    
    2. Greater integration with other technologies
    


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
    Generated text:  [Name]. I'm a [job title] with [number of years in the industry] years of experience in [the field]. I'm comfortable and enthusiastic about [industry]. I'm a [industry] specialist, with a passion for [industry] and a love for [industry]. I'm looking to grow my skills and knowledge in this field, and I'm excited to share my experiences and insights with others. I'm always learning and seeking new challenges to help me grow professionally. 
    
    Any last words of wisdom for new professionals in the [industry]? 
    
    I hope you find my profile helpful! Let me know if you need any
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France, the capital of France, and the second most populous city in the European Union. It is the seat of the French government and the largest city in metropolitan France. It is also home to the headquarters of many of France's major international organizations and the seat of the French National Assembly. The city is known for its rich history, diverse culture, and stunning architecture. Paris is famous for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also the home of the French revolution, where the first French Republic was proclaimed in 17
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be dominated by a wide range of emerging technologies and applications that will continue to reshape industries and transform human life. Some of the key trends in AI include:
    
    1. Increased automation and robotics: As automation and robotics continue to improve, we may see more machines performing tasks that were previously done by humans, such as manufacturing, service, and administrative tasks. This could lead to significant job displacement and create new opportunities for those who can adapt and learn to work in these new roles.
    
    2. Enhanced data analytics and machine learning: With the increasing amount of data available to us, AI will be able to learn from it and make more accurate


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

     Emily

    .

     I

    'm

     a

     

    3

    5

    -year

    -old

     freelance

     writer

     who

     specializes

     in

     children

    's

     literature

    .

     I

     love

     to

     write

     stories

     that

     engage

     readers

     and

     bring

     joy

     to

     families

    .

     My

     writing

     style

     is

     straightforward

     and

     rel

    atable

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     share

     my

     ideas

     with

     others

    .

     I

    'm

     always

     up

     for

     a

     challenge

     and

     enjoy

     sharing

     my

     work

     with

     my

     friends

     and

     family

    .

     Can

     you

     tell

     me

     more

     about

     your

     writing

     style

     and

     how

     you

     plan

     to

     use

     it

     in

     your

     future

     projects

    ?


    Emily

    ,

     you

     are

     a

     talented

     writer

     with

     a

     passion

     for

     children

    's

     literature

    .

     How

     do

     you

     plan

     to

     use

     your

     writing

     skills

     in

     your

     future

     projects

    ?


    Emily

    ,

     thank

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     city

     with

     the

     most

     population

     and

     the

     second

     largest

     by

     area

    ,

     the

     

    1

    2

    th

     in

     the

     world

    .

     Its

     population

     is

     around

     

    2

    1

     million

     as

     of

     

    2

    0

    2

    1

    .

     Paris

     is

     the

     seat

     of

     the

     Government

     of

     France

    ,

     and

     a

     major

     met

    ropolis

    ,

     and

     has

     been

     since

     

    1

    7

    9

    0

    .

     Paris

     is

     famous

     for

     its

     architecture

    ,

     art

    ,

     and

     culture

    ,

     as

     well

     as

     for

     its

     annual

     Car

    ême

     celebrations

    ,

     which

     are

     associated

     with

     Christmas

    .

     Paris

     is

     the

     cultural

     and

     political

     center

     of

     France

    ,

     and

     one

     of

     the

     world

    's

     most

     visited

     cities

    .

     It

     is

     a

     popular

     tourist

     destination

    ,

     and

     is

     home

     to

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     more

     diverse

    ,

     interconnected

    ,

     and

     personalized

     than

     it

     is

     today

    .

     Some

     of

     the

     possible

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     integration

    :

     AI

     will

     continue

     to

     be

     integrated

     into

     various

     sectors

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     agriculture

    .

     This

     integration

     will

     drive

     new

     applications

     and

     opportunities

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Self

    -driving

     cars

    ,

     drones

    ,

     and

     robots

     are

     expected

     to

     become

     more

     prevalent

     in

     the

     future

    .

     These

     vehicles

     will

     be

     able

     to

     navigate

     on

     their

     own

    ,

     reducing

     the

     need

     for

     human

     drivers

    .
    


    3

    .

     Personal

    ized

     healthcare

    :

     AI

     will

     allow

     doctors

     to

     provide

     more

     accurate

     and

     personalized

     medical

     advice

     to

     patients

    .

     AI

     will

     also

     help

     in

     developing

     new

    



```python
llm.shutdown()
```
