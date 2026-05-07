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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.31it/s]


    2026-05-07 18:50:24,954 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 18:50:24] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.61it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.30it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.30it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.30it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.30it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.30it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.30it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.30it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.30it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.30it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.30it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.55it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.88it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.88it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.88it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.88it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.88it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.88it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.88it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.88it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.88it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.88it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.88it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.13it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.13it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.13it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.13it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.13it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.13it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.13it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.23 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.20 GB):   3%|▎         | 2/58 [00:00<00:02, 19.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.20 GB):   3%|▎         | 2/58 [00:00<00:02, 19.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.19 GB):   3%|▎         | 2/58 [00:00<00:02, 19.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.19 GB):   3%|▎         | 2/58 [00:00<00:02, 19.63it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.87it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.87it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.18 GB):   9%|▊         | 5/58 [00:00<00:02, 22.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.17 GB):   9%|▊         | 5/58 [00:00<00:02, 22.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.17 GB):   9%|▊         | 5/58 [00:00<00:02, 22.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=57.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.15 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.15 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.15 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.14 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.14 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.14 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.14 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s]Capturing num tokens (num_tokens=960 avail_mem=57.13 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s] Capturing num tokens (num_tokens=896 avail_mem=57.13 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s]

    Capturing num tokens (num_tokens=832 avail_mem=57.13 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s]Capturing num tokens (num_tokens=832 avail_mem=57.13 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.57it/s]Capturing num tokens (num_tokens=768 avail_mem=57.12 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.57it/s]Capturing num tokens (num_tokens=704 avail_mem=57.12 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.57it/s]Capturing num tokens (num_tokens=640 avail_mem=57.12 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.57it/s]Capturing num tokens (num_tokens=576 avail_mem=57.12 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.57it/s]Capturing num tokens (num_tokens=512 avail_mem=57.10 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.57it/s]Capturing num tokens (num_tokens=512 avail_mem=57.10 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=480 avail_mem=57.12 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=448 avail_mem=57.12 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=416 avail_mem=57.11 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=384 avail_mem=57.11 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]

    Capturing num tokens (num_tokens=352 avail_mem=57.11 GB):  50%|█████     | 29/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=352 avail_mem=57.11 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.25it/s]Capturing num tokens (num_tokens=320 avail_mem=57.10 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.25it/s]Capturing num tokens (num_tokens=288 avail_mem=57.10 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.25it/s]Capturing num tokens (num_tokens=256 avail_mem=56.82 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.25it/s]Capturing num tokens (num_tokens=240 avail_mem=56.81 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.25it/s]Capturing num tokens (num_tokens=224 avail_mem=56.81 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.25it/s]Capturing num tokens (num_tokens=224 avail_mem=56.81 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=208 avail_mem=56.80 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=192 avail_mem=56.80 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=176 avail_mem=56.80 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.51it/s]

    Capturing num tokens (num_tokens=160 avail_mem=56.80 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=144 avail_mem=56.79 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=144 avail_mem=56.79 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=128 avail_mem=56.79 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=112 avail_mem=56.79 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=96 avail_mem=56.79 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s] Capturing num tokens (num_tokens=80 avail_mem=56.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=64 avail_mem=56.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=64 avail_mem=56.78 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.46it/s]Capturing num tokens (num_tokens=48 avail_mem=56.77 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.46it/s]Capturing num tokens (num_tokens=32 avail_mem=56.77 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.46it/s]Capturing num tokens (num_tokens=28 avail_mem=56.77 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.46it/s]

    Capturing num tokens (num_tokens=24 avail_mem=56.76 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.46it/s]Capturing num tokens (num_tokens=20 avail_mem=56.76 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.46it/s]Capturing num tokens (num_tokens=20 avail_mem=56.76 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=16 avail_mem=56.76 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=12 avail_mem=56.75 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=8 avail_mem=56.75 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.16it/s] Capturing num tokens (num_tokens=4 avail_mem=56.75 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=4 avail_mem=56.75 GB): 100%|██████████| 58/58 [00:01<00:00, 41.46it/s]


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
    Generated text:  Michelle and I am the Marketing Manager at the Lake Orion Fire Department. My role includes marketing and sales for the department and assisting the department with their budget and budget variances.
    I am currently an active duty employee of the Canadian Armed Forces. I am currently assigned to the 2nd Battalion, 45th Armor Regiment. My most recent assignment is with the 304th Parachute Regiment.
    Michelle’s current responsibilities include marketing, sales, and budget management for the Department. She also assists the Department with its budget and budget variances. Michelle is responsible for developing, implementing, and evaluating marketing strategies, as well as
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide between two jobs. He has a 50% chance of getting a first job offer and a 50% chance of getting a second job offer. The first job offer comes with a guaranteed annual salary of $25,000,000, while the second job offer comes with a guaranteed annual salary of $30,000,000. The president is concerned about the potential for losing the job at any time. He has a 20% chance of losing the first job offer and a 30% chance of losing the second job offer. If the president
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris  
    B. Rome  
    C. Berlin  
    D. Ottawa
    
    1. **Identify the question:** Determine which city in the given list is the capital of France.
    
    2. **List of cities in the list:** Paris, Rome, Berlin, Ottawa.
    
    3. **Determine the capital of France:** 
    
       - Paris is the capital of France, as it is the largest city in the country.
       - Rome is the capital of Italy, not France.
       - Berlin is the capital of Germany, not France.
       - Ottawa is the capital of Canada, not France.
    
    4. **Conclusion:** The
    ===============================
    Prompt: The future of AI is
    Generated text:  big and it is definitely real. As a great AI architect, I need to plan for the future of AI. How can I achieve this?
    The future of AI is vast and multifaceted, encompassing the fields of robotics, machine learning, natural language processing, computer vision, and more. To stay up to date on the latest developments in AI, it's essential to keep abreast of the latest advancements in each area. Additionally, it's important to stay updated on the latest trends and innovations in each area of AI.
    To achieve this, I can follow these steps:
    
    1. Stay current on the latest advancements in each area


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is also home to the French Parliament and the French National Library. The French capital is a vibrant and dynamic city with a rich cultural heritage. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is also known for its diverse population, with many different
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be increased emphasis on ethical and social considerations. This could lead to more transparent and accountable AI systems that are designed to minimize harm and maximize benefits.
    
    3. Increased use of AI in healthcare: AI is likely to play a larger role in
    


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
    Generated text:  [name], and I work at [company]. I'm here to introduce myself and ask you to introduce yourself. Let's get started. Sure, I'd be happy to do that! My name is [name], and I'm here to introduce myself and ask you to introduce yourself. What's your name, and what company do you work for? That's helpful information. How can I help you learn more about [company]? Let me know how I can assist you! I'm excited to have the opportunity to learn more about [company] and get to know you. How can I help you learn more about [company]?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the largest city and one of the most populous urban areas in the world. It is located on the Seine river, near the outskirts of the Paris basin, and serves as the administrative, cultural, and economic center of France. Paris is known for its beautiful architecture, rich history, and delicious cuisine, as well as its rich cultural heritage and art scene. The city has a diverse population of over 2 million people and is home to numerous museums, theaters, and other cultural institutions. Paris is often referred to as the "city of love" and is considered one of the most beautiful cities in the world. Despite
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bound to be shaped by a number of trends, including:
    
    1. Increased focus on ethical AI: As the industry moves closer to the point where AI is increasingly integrated into our daily lives, the need for a focus on ethical AI is likely to increase. This could mean increased scrutiny of AI algorithms and the data they use, as well as greater use of transparency and accountability in AI decision-making.
    
    2. Greater use of machine learning and deep learning: As AI becomes more integrated into our daily lives, there will be an increasing need for machines to learn and adapt to new situations. This could mean greater use of machine learning and deep learning,


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

     a

     [

    job

     title

    ]

     at

     [

    company

     name

    ].

     I

     have

     a

     passion

     for

     [

    specific

     interest

     or

     hobby

    ]

     and

     have

     been

     [

    number

     of

     years

     in

     this

     role

    ]

     in

     this

     position

    .

     I

     love

     being

     surrounded

     by

     people

     who

     appreciate

     my

     unique

     style

     and

     are

     willing

     to

     listen

     to

     me

    .

     Please

     let

     me

     know

     if

     you

    're

     interested

     in

     learning

     more

     about

     me

     or

     if

     there

    's

     anything

     I

     can

     do

     to

     help

     you

     out

    .

     [

    Name

    ]

     [

    Company

     name

    ]

     [

    Date

     of

     last

     interaction

    ]

     [

    Company

     URL

     if

     available

    ]

     [

    Company

     Twitter

     if

     available

    ]

     [

    Company

     LinkedIn

     if

     available

    ]

     [

    Company

     Instagram

     if

     available

    ]

     [

    Company

     Facebook

     if

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     often

     referred

     to

     as

     the

     "

    City

     of

     Love

    ,"

     is

     the

     capital

     city

     of

     the

     country

     and

     is

     located

     in

     the

     region

     of

     Î

    le

    -de

    -F

    rance

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     land

     area

     and

     the

     third

    -largest

     city

     by

     population

    .

     It

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     its

     museums

     and

     art

     galleries

    ,

     and

     its

     bustling

     street

     life

    .

     Paris

     is

     also

     famous

     for

     its

     romantic

     architecture

     and

     the

     city

    's

     love

     of

     wine

     and

     gastr

    onomy

    .

     The

     city

     is

     a

     major

     economic

     and

     cultural

     center

     and

     is

     home

     to

     many

     prestigious

     universities

     and

     institutions

    .

     Paris

     has

     a

     rich

     history

     and

     is

     a

     significant

     cultural

     and

     political

     hub

     in

     Western

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     key

     trends

     that

     will

     shape

     its

     development

     and

     impact

     on

     various

     sectors

    .

     Here

     are

     some

     potential

     trends

     that

     are

     expected

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     As

     AI

     continues

     to

     advance

    ,

     it

     is

     likely

     to

     become

     more

     integrated

     with

     human

     intelligence

    .

     This

     integration

     could

     lead

     to

     more

     complex

     and

     creative

     AI

     systems

    ,

     as

     well

     as

     more

     intelligent

     and

     empath

    etic

     AI

     agents

    .

     This

     trend

     could

     be

     seen

     in

     the

     development

     of

     systems

     that

     can

     understand

     and

     respond

     to

     human

     emotions

     and

     preferences

    .
    


    2

    .

     Greater

     use

     of

     AI

     in

     healthcare

    :

     As

     AI

     becomes

     more

     integrated

     with

     human

     intelligence

    ,

     it

     is

     likely

     to

     play

     an

     increasingly

    



```python
llm.shutdown()
```
