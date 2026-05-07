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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.18it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.17it/s]


    2026-05-07 14:09:41,290 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 14:09:41] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:39,  4.90s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:39,  4.90s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:39,  4.90s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:39,  4.90s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:39,  4.90s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.60it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.60it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.60it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.60it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.60it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.60it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.60it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.60it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.60it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.60it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.91it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.91it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.91it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.91it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.91it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.91it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.91it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.91it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 12.91it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 18.73it/s]

    Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 25.52it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 25.52it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 25.52it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 25.52it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 25.52it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 25.52it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 25.52it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 25.52it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 25.52it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 32.69it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 32.69it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 32.69it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 32.69it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 32.69it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 32.69it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 32.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:03, 18.01it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:03, 18.01it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.18 GB):   3%|▎         | 2/58 [00:00<00:03, 18.01it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.18 GB):   3%|▎         | 2/58 [00:00<00:03, 18.01it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.18 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.12 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.89 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.88 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.88 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.88 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.73it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=73.88 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.87 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.87 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.87 GB):  21%|██        | 12/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.87 GB):  21%|██        | 12/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.87 GB):  21%|██        | 12/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.86 GB):  21%|██        | 12/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.86 GB):  21%|██        | 12/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.86 GB):  21%|██        | 12/58 [00:00<00:01, 26.75it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=73.86 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.85 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.85 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.85 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.83it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.83 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.83it/s]Capturing num tokens (num_tokens=960 avail_mem=73.84 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.83it/s] Capturing num tokens (num_tokens=960 avail_mem=73.84 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.87it/s]Capturing num tokens (num_tokens=896 avail_mem=73.84 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.87it/s]Capturing num tokens (num_tokens=832 avail_mem=73.84 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.87it/s]Capturing num tokens (num_tokens=768 avail_mem=73.83 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.87it/s]Capturing num tokens (num_tokens=704 avail_mem=73.83 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.87it/s]

    Capturing num tokens (num_tokens=640 avail_mem=73.83 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.87it/s]Capturing num tokens (num_tokens=640 avail_mem=73.83 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.96it/s]Capturing num tokens (num_tokens=576 avail_mem=73.83 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.96it/s]Capturing num tokens (num_tokens=512 avail_mem=73.81 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.96it/s]Capturing num tokens (num_tokens=480 avail_mem=73.83 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.96it/s]Capturing num tokens (num_tokens=448 avail_mem=73.82 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.96it/s]Capturing num tokens (num_tokens=416 avail_mem=73.82 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.96it/s]Capturing num tokens (num_tokens=416 avail_mem=73.82 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.37it/s]Capturing num tokens (num_tokens=384 avail_mem=73.82 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.37it/s]Capturing num tokens (num_tokens=352 avail_mem=73.82 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.37it/s]Capturing num tokens (num_tokens=320 avail_mem=73.81 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.37it/s]

    Capturing num tokens (num_tokens=288 avail_mem=73.81 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.37it/s]Capturing num tokens (num_tokens=256 avail_mem=73.81 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.37it/s]Capturing num tokens (num_tokens=256 avail_mem=73.81 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=240 avail_mem=73.80 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=224 avail_mem=73.80 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=208 avail_mem=73.79 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=192 avail_mem=73.79 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=176 avail_mem=73.79 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=176 avail_mem=73.79 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=160 avail_mem=73.79 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=144 avail_mem=73.78 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s]

    Capturing num tokens (num_tokens=128 avail_mem=73.78 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=112 avail_mem=73.78 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=96 avail_mem=73.78 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.07it/s] Capturing num tokens (num_tokens=96 avail_mem=73.78 GB):  81%|████████  | 47/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=80 avail_mem=73.77 GB):  81%|████████  | 47/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=64 avail_mem=73.77 GB):  81%|████████  | 47/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=48 avail_mem=73.76 GB):  81%|████████  | 47/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=32 avail_mem=73.76 GB):  81%|████████  | 47/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=28 avail_mem=73.76 GB):  81%|████████  | 47/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=28 avail_mem=73.76 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.22it/s]Capturing num tokens (num_tokens=24 avail_mem=73.75 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.22it/s]

    Capturing num tokens (num_tokens=20 avail_mem=73.75 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.22it/s]Capturing num tokens (num_tokens=16 avail_mem=73.75 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.22it/s]Capturing num tokens (num_tokens=12 avail_mem=73.74 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.22it/s]Capturing num tokens (num_tokens=8 avail_mem=73.74 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.22it/s] Capturing num tokens (num_tokens=8 avail_mem=73.74 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=4 avail_mem=73.74 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=4 avail_mem=73.74 GB): 100%|██████████| 58/58 [00:01<00:00, 35.98it/s]


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
    Generated text:  Dara.
    
    I have been working as a student abroad in the U.S. for the past 2 years. I'm not sure what my future plans are, but I do have some ideas. I think that I want to learn more about how to work as a junior member of the U.S. Navy SEAL Team 6. 
    
    I also have some questions about different job functions in the Navy. I think I might be able to help someone in a job function that I have some experience with, but I don't know what that would be.
    
    I was wondering if you could help me with a question like this. I want to
    ===============================
    Prompt: The president of the United States is
    Generated text:  30 years older than the president of Brazil. The president of Brazil is 25 years younger than the president of France. If the president of France is currently 40 years old, how old would the president of Brazil be in 5 years? Let's solve the problem step by step.
    
    1. **Determine the age of the president of Brazil:**
       - The president of France is currently 40 years old.
       - The president of Brazil is 25 years younger than the president of France.
       - Therefore, the president of Brazil is:
         \[
         40 - 2
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Nice
    C. Marseille
    D. Lille
    Answer:
    
    A
    
    The next sentence in the text should be:
    A. The capital of France is Paris.
    B. The capital of France is Nice.
    C. The capital of France is Marseille.
    D. The capital of France is Lille.
    
    Answer:
    
    A
    
    In this problem, the correct answer is:
    
    Answer:
    
    A
    
    In the given passage, which of the following is the correct order of sentences?
    
    A. ① The capital of France is Paris.
    ② The capital of France is Nice.
    ③ The capital of
    ===============================
    Prompt: The future of AI is
    Generated text:  moving from its current mode of being used to generate machine learning models to being used in a more holistic way to improve the entire product development lifecycle. This transition is happening due to the increasing demand for automation and automation of processes in the product development process, and the improvement of the existing technologies and systems used in the process. In 2019, for example, it was noted that about 75% of the development processes used in the industry were experiencing a significant increase in complexity. Hence, to meet the increasing demand for automation and automation of processes in the product development process, it is crucial to integrate AI technologies into the product


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also known for its cuisine, including its famous French fries and its traditional French wine. The city is also home to a diverse population of people from all over the world, making it a vibrant and multicultural city. Paris is a city of contrasts and is a must-visit destination for anyone
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve the quality of care. As AI technology continues to advance, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in
    


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
    Generated text:  [Name] and I'm a [job title] who has a passion for [job title] and [occupation]. I’m driven, energetic, and always looking for opportunities to learn and grow in my field. I believe in the power of teamwork and leadership, and I’m excited to bring my skills and experiences to any team I'm a part of. I’m always up for a challenge and eager to grow and improve my abilities. I'm confident in my ability to contribute to the success of any team I work with, and I'm excited to work with [job title] on [project or team].
    Sincerely,
    [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the second largest city in the European Union and the sixth-largest city in the world by population, after Beijing and Moscow. Paris has been a political, cultural, and economic center of Europe since the 13th century. It was the capital of France from 1792 to 1976, and from 1989 to 2004. It is also the most visited city in the world according to the World Travel and Tourism Council. It is known for its art, architecture, and food, and for its scenic landscapes, including the Louvre and the Eiffel
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid development and innovation, with a focus on increasing the ability of machines to understand and interact with humans in increasingly complex and nuanced ways. Some potential future trends in AI include:
    
    1. Increased focus on ethics and responsible AI: As the technology becomes more integrated into society, there will be increasing pressure on developers and users to consider the broader implications of AI, including issues such as bias, fairness, and accountability.
    
    2. Continued development of advanced AI systems: AI systems will continue to get better at performing tasks that require complex reasoning and decision-making, such as image and speech recognition, natural language processing, and decision-making


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

    ],

     and

     I

    'm

     [

    Age

    ].

     I

    'm

     a

     [

    job

     title

    ]

     and

     I

     have

     been

     in

     the

     field

     for

     [

    number

     of

     years

    ].

     I

    've

     always

     been

     passionate

     about

     [

    career

     goal

     or

     interest

    ],

     and

     I

    'm

     determined

     to

     achieve

     it

    .

     Whether

     it

    's

     through

     [

    specific

     activity

    ,

     e

    .g

    .

     tutoring

    ,

     writing

    ,

     etc

    .

    ],

     I

    'm

     always

     ready

     to

     learn

     and

     grow

    .

     I

     enjoy

     [

    other

     interest

     or

     hobby

    ],

     and

     I

    'm

     always

     looking

     for

     opportunities

     to

     meet

     and

     support

     people

     who

     share

     my

     interests

    .

     I

    'm

     a

     team

     player

    ,

     and

     I

     thrive

     in

     a

     collaborative

     environment

    .

     I

    'm

     always

     open

     to

     new

     experiences

     and

     I

    'm

     excited

     to

     learn

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     vast

     and

     exciting

    ,

     with

     many

     potential

     developments

     and

     areas

     of

     interest

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     use

     of

     AI

     for

     autonomous

     vehicles

    :

     Autonomous

     vehicles

     will

     become

     more

     prevalent

     as

     AI

     becomes

     more

     advanced

     and

     reliable

    .

     This

     will

     lead

     to

     a

     reduction

     in

     the

     need

     for

     human

     drivers

    ,

     allowing

     for

     a

     safer

     and

     more

     efficient

     transportation

     system

    .
    


    2

    .

     AI

     in

     healthcare

    :

     AI

     will

     be

     used

     to

     improve

     the

     accuracy

     and

     efficiency

     of

     medical

     diagnostics

    ,

     treatment

     planning

    ,

     and

     patient

     care

    .

     This

     will

     lead

     to

     more

     personalized

     treatment

     options

     and

     better

     outcomes

     for

     patients

    .
    


    3

    .

     AI

     in

     finance

    :

     AI

     will

     be

     used

     to

     improve

     risk

     management

     and

    



```python
llm.shutdown()
```
