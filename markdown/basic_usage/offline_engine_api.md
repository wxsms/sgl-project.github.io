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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.26it/s]


    2026-05-14 00:39:51,041 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 00:39:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.48it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]

    Compiling num tokens (num_tokens=416):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=176):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=160):  55%|█████▌    | 32/58 [00:04<00:01, 16.85it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:04<00:00, 25.81it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:04<00:00, 25.81it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:04<00:00, 25.81it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:04<00:00, 25.81it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:04<00:00, 25.81it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:04<00:00, 25.81it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:04<00:00, 25.81it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:04<00:00, 25.81it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:04<00:00, 25.81it/s]

    Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:04<00:00, 25.81it/s]Compiling num tokens (num_tokens=24):  74%|███████▍  | 43/58 [00:04<00:00, 25.81it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:04<00:00, 34.88it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:04<00:00, 34.88it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 34.88it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 34.88it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 34.88it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 34.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.84 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.81 GB):   3%|▎         | 2/58 [00:00<00:03, 15.19it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.81 GB):   3%|▎         | 2/58 [00:00<00:03, 15.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.81 GB):   3%|▎         | 2/58 [00:00<00:03, 15.19it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.81 GB):   7%|▋         | 4/58 [00:00<00:03, 17.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.81 GB):   7%|▋         | 4/58 [00:00<00:03, 17.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.80 GB):   7%|▋         | 4/58 [00:00<00:03, 17.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.79 GB):   7%|▋         | 4/58 [00:00<00:03, 17.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.79 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.79 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.79 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.78 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.42it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=71.78 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.78 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.77 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.77 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.77 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.76 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.76 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.76 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.76 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.78it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.75 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.78it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.75 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.75 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.78it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=71.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.78it/s]Capturing num tokens (num_tokens=960 avail_mem=71.74 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.78it/s] Capturing num tokens (num_tokens=960 avail_mem=71.74 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=896 avail_mem=71.74 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=832 avail_mem=71.74 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=768 avail_mem=71.73 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=704 avail_mem=71.42 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=704 avail_mem=71.42 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.80it/s]Capturing num tokens (num_tokens=640 avail_mem=71.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.80it/s]

    Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.80it/s]Capturing num tokens (num_tokens=512 avail_mem=71.44 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.80it/s]Capturing num tokens (num_tokens=480 avail_mem=71.47 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.80it/s]Capturing num tokens (num_tokens=480 avail_mem=71.47 GB):  52%|█████▏    | 30/58 [00:00<00:00, 31.61it/s]Capturing num tokens (num_tokens=448 avail_mem=71.48 GB):  52%|█████▏    | 30/58 [00:00<00:00, 31.61it/s]Capturing num tokens (num_tokens=416 avail_mem=71.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.61it/s]

    Capturing num tokens (num_tokens=384 avail_mem=71.67 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.61it/s]Capturing num tokens (num_tokens=352 avail_mem=71.66 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.61it/s]Capturing num tokens (num_tokens=352 avail_mem=71.66 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.41it/s]Capturing num tokens (num_tokens=320 avail_mem=71.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.41it/s]Capturing num tokens (num_tokens=288 avail_mem=71.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.41it/s]Capturing num tokens (num_tokens=256 avail_mem=71.64 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.41it/s]Capturing num tokens (num_tokens=240 avail_mem=71.64 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.41it/s]

    Capturing num tokens (num_tokens=240 avail_mem=71.64 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.61it/s]Capturing num tokens (num_tokens=224 avail_mem=71.63 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.61it/s]Capturing num tokens (num_tokens=208 avail_mem=71.62 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.61it/s]Capturing num tokens (num_tokens=192 avail_mem=71.62 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.61it/s]Capturing num tokens (num_tokens=176 avail_mem=71.61 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.61it/s]Capturing num tokens (num_tokens=176 avail_mem=71.61 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=160 avail_mem=71.61 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=144 avail_mem=71.60 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=128 avail_mem=71.59 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.79it/s]Capturing num tokens (num_tokens=112 avail_mem=71.59 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.79it/s]

    Capturing num tokens (num_tokens=112 avail_mem=71.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.75it/s]Capturing num tokens (num_tokens=96 avail_mem=71.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.75it/s] Capturing num tokens (num_tokens=80 avail_mem=71.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.75it/s]Capturing num tokens (num_tokens=64 avail_mem=71.55 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.75it/s]Capturing num tokens (num_tokens=48 avail_mem=71.55 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.75it/s]Capturing num tokens (num_tokens=48 avail_mem=71.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=32 avail_mem=71.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=28 avail_mem=71.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=24 avail_mem=71.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=20 avail_mem=71.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=16 avail_mem=71.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.51it/s]

    Capturing num tokens (num_tokens=16 avail_mem=71.52 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.70it/s]Capturing num tokens (num_tokens=12 avail_mem=71.51 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.70it/s]Capturing num tokens (num_tokens=8 avail_mem=71.51 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.70it/s] Capturing num tokens (num_tokens=4 avail_mem=71.50 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.70it/s]Capturing num tokens (num_tokens=4 avail_mem=71.50 GB): 100%|██████████| 58/58 [00:01<00:00, 32.24it/s]


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
    Generated text:  Max, and I'm a computer science student. I was born on November 10, 1997, and I'm in the third year of my program. I'm currently attending University of Illinois at Urbana-Champaign.
    
    I am particularly interested in data science, and my research interest is in the field of recommendation systems. I've recently started using the recommender system library in Python to develop my own recommendation system for a university course. The system is able to provide personalized recommendations for students based on their preferences.
    
    The current system has a few issues, such as heavy computation time and low accuracy of recommendations. I want
    ===============================
    Prompt: The president of the United States is
    Generated text:  a male. Among the following, the one who is a female is (　　)
    A: The president of the United States  
    B: The member of Congress  
    C: The Secretary of State of the United States  
    D: The deputy Speaker of the House of Representatives To determine which of the given options is a female, we need to consider the roles and positions of the U.S. President, members of Congress, the Secretary of State, and the deputy Speaker of the House of Representatives.
    
    1. The President of the United States is a male.
    2. Members of Congress are elected officials in the House of Representatives, which are
    ===============================
    Prompt: The capital of France is
    Generated text:  _______ ( )
    
    A: Paris  
    B: London  
    C: Rome  
    D: Moscow
    
    To determine the capital of France, let's analyze the options provided:
    
    A: Paris - This is the capital of France. Paris is known for its iconic Notre-Dame Cathedral and is a major cultural center.
    
    B: London - This is the capital of the United Kingdom. While Paris is indeed near London, it is not the capital of France.
    
    C: Rome - This is the capital of Italy. While Rome is a significant city, it is not the capital of France.
    
    D: Moscow - This is the capital of Russia. While
    ===============================
    Prompt: The future of AI is
    Generated text:  very bright, but it's also very complex. Understanding how AI algorithms operate, making them work, and ensuring that they are used safely is essential. To develop a robust and secure AI system, you need to start by understanding the underlying concepts and principles that govern how AI works. This is where a comprehensive understanding of machine learning and computer science is important. You should also familiarize yourself with the regulations and guidelines that govern the use of AI systems in various industries.
    The next step is to learn about the technical details of how AI algorithms are implemented and deployed. This requires a deep understanding of programming languages, data structures, and data processing techniques


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your profession or experience here]. I enjoy [insert a short description of your hobbies or interests here]. What do you do for a living? I'm always looking for new challenges and opportunities to grow and learn. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, music, and fashion, and is home to many world-renowned museums, theaters, and restaurants. The city is also known for its vibrant nightlife and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city of people, with its diverse population and culture making it a unique
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into our daily lives, from home automation to autonomous vehicles. As AI becomes more capable of performing tasks that were previously done by humans, we may see an increase in automation in various industries.
    
    2. AI ethics and privacy: As AI becomes more integrated into our lives, there will be a need to address the ethical and privacy concerns associated with AI. This will likely lead to increased regulation and oversight of AI development and deployment.
    
    3
    


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
    Generated text:  [insert name]. I'm a [insert age range], [insert gender] and I am [insert occupation]. I'm a software engineer [insert profession] with a master's degree in [insert degree]. I'm passionate about [insert passion or interest]. My interests include [insert hobbies or activities]. I am [insert unique personality trait or background]. I'm [insert website or LinkedIn profile] and I enjoy [insert activities or interests]. I'm a [insert fictional role or avatar] and I'm dedicated to [insert purpose or mission]. I believe in [insert values or principles]. I'm excited to [insert future goals or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The statement provided is factual and directly addresses the question of what the capital city of France is. It is a complete statement that includes all the key information needed to provide the requested information. 
    
    Therefore, the answer is:
    
    Paris. 
    
    This statement encapsulates the core facts about the capital city of France. 
    
    The key points to remember:
    - It is the capital city of France.
    - Its full name is Paris. 
    
    The statement "The capital of France is Paris" accurately and concisely describes the capital city of France. Therefore, the final answer is:
    \boxed{Paris}
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but it is expected to continue to evolve rapidly, with many potential trends developing in the coming decades. Here are some of the possible future trends in AI:
    
    1. Deep Learning: As the technology of deep learning continues to advance, AI is expected to become more capable of solving complex problems and making decisions that are more accurate and efficient. Deep learning algorithms will be able to learn from large and complex datasets much faster than traditional machine learning algorithms.
    
    2. AI Ethics: As AI technology continues to advance, it is expected to face new ethical challenges, such as privacy, bias, and accountability. There is a growing recognition that AI systems


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

     ______

    .

     My

     goal

     is

     to

     help

     people

     with

     various

     issues

    .

     I

    'm

     always

     trying

     to

     make

     people

     feel

     better

    .

     I

    'm

     confident

     that

     with

     my

     skills

    ,

     I

     can

     help

     people

     achieve

     their

     goals

    .

     I

    'm

     available

     

    2

    4

    /

    7

     and

     can

     be

     reached

     at

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    __

    _.

     My

     interests

     are

     __

    ________

    _

     and

     I

     enjoy

     __

    ________

    _

    .
    


    Hello

    ,

     my

     name

     is

     __

    ________

    .

     I

     want

     to

     help

     people

     with

     various

     issues

    .

     I

    'm

     always

     trying

     to

     make

     people

     feel

     better

    .

     I

    'm

     confident

     that

     with

     my

     skills

    ,

     I

     can

     help

     people

     achieve

     their

     goals

    .

     I

    'm

     available

     

    2

    4

    /

    7

     and

     can

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     on

     the

     Se

    ine

     river

    ,

     in

     the

     south

     of

     the

     country

    .

     The

     city

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

     It

     is

     the

     world

    's

     largest

     metropolitan

     area

     and

     is

     home

     to

     many

     of

     the

     country

    's

     most

     famous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     a

     major

     center

     for

     education

    ,

     arts

    ,

     and

     media

    ,

     and

     its

     city

     life

     is

     characterized

     by

     its

     embrace

     of

     modern

    ity

     and

     its

     commitment

     to

     preserving

     historical

     and

     cultural

     traditions

    .

     It

     is

     the

     seat

     of

     the

     French

     government

     and

     the

     largest

     economy

     in

     the

     European

     Union

    ,

     and

     remains

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     constantly

     evolving

    ,

     with

     numerous

     potential

     areas

     of

     innovation

     and

     advancement

    .

     Here

     are

     some

     possible

     trends

     in

     AI

    :
    


    1

    .

     Increased

     emphasis

     on

     ethical

     considerations

    :

     As

     AI

     systems

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     ethical

     considerations

     and

     safeguards

    .

     We

     will

     need

     to

     ensure

     that

     AI

     systems

     are

     designed

     and

     used

     in

     a

     way

     that

     respects

     privacy

    ,

     protects

     personal

     information

    ,

     and

     minim

    izes

     harm

     to

     individuals

    .
    


    2

    .

     Improved

     accuracy

     and

     precision

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     we

     will

     see

     a

     significant

     increase

     in

     the

     accuracy

     and

     precision

     of

     its

     predictions

     and

     decisions

    .

     This

     will

     require

     the

     development

     of

     new

     algorithms

     and

     techniques

     for

     statistical

     and

     machine

     learning

    



```python
llm.shutdown()
```
