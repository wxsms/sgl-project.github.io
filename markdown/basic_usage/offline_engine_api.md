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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.37it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.36it/s]


    2026-04-06 03:23:10,877 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 03:23:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<02:45,  2.91s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:08,  5.77it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:08,  5.77it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:08,  5.77it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:08,  5.77it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:08,  5.77it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:08,  5.77it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:08,  5.77it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:08,  5.77it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:08,  5.77it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 11.74it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 11.74it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 11.74it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 11.74it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 11.74it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 11.74it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 11.74it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 11.74it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 23.97it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 23.97it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 23.97it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 23.97it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 23.97it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 23.97it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 23.97it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 28.95it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 28.95it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 28.95it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 28.95it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 28.95it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 28.95it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 28.95it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 33.86it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 33.86it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 33.86it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 33.86it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 33.86it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 33.86it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 33.86it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=134.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=132.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=132.63 GB):   3%|▎         | 2/58 [00:00<00:03, 16.59it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 16.59it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 16.59it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 16.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.39 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.59it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s]Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.05it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.05it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.05it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.05it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.05it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.05it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.17it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.17it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.17it/s]

    Capturing num tokens (num_tokens=448 avail_mem=120.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.17it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.17it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.17it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.67it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.67it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.67it/s]Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.67it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.67it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.54it/s]

    Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.16it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.16it/s]

    Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.83it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.83it/s]

    Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 44.58it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 38.94it/s]


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
    Generated text:  Xiaohong and I am a medical student. This is my first visit to the emergency department (ED). I am a sophomore at a college, majoring in medicine. I have been admitted to the hospital and went through all the necessary health check-ups, including a routine physical examination, blood test, and allergy test.
    The doctors informed me that I have a history of severe asthma and are being treated with long-acting beta agonists (LABAs), salbutamol, and a corticosteroid inhaler (ICS) and that I was prescribed a prescription for a post-nasal drip humidifier. As I examined
    ===============================
    Prompt: The president of the United States is
    Generated text:  200 years old. If you subtract 10 years from the current year, how old will the president be?
    To determine the age of the president when you subtract 10 years from the current year, we need to follow these steps:
    
    1. Identify the current year.
    2. Subtract 10 years from the current year.
    
    Since the current year is not specified in the problem, let's assume the current year is 2023 for the sake of this calculation. If the current year is different, we can substitute that value.
    
    Step-by-step solution:
    
    1. Identify the current year: The current
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. This is the only city that has served as both its own capital and a capital city of the United Kingdom, both of which are in Europe. Paris has been the capital city of France since the beginning of the 12th century. The city was founded by the Romans and was an important stop on the Via Erythraea (the Erythraean road) which linked the Mediterranean to the Atlantic.
    
    If you go to Paris, you are likely to see the Louvre, the Eiffel Tower, the Palace of Versailles, the Arc de Triomphe, and the Notre Dame Cathedral. Each
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it is not without its challenges. One challenge is the bias problem, where machines tend to produce results that are biased against certain groups of people. Bias can occur in AI systems when the data used to train the AI is not representative of the broader population, or when the AI is designed with an unconscious bias that is not reflected in the data it was trained on.
    To combat bias, companies are using a range of techniques to ensure that their AI systems are fair and unbiased. One approach is to use models that are trained on diverse data sets, ensuring that they are not biased towards any particular group. Another approach is to use


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


    Generated text:  Paris, known for its iconic Eiffel Tower and rich history dating back to the 12th century. It is also home to the Louvre Museum, the most famous art museum in the world, and the Notre-Dame Cathedral, a Gothic masterpiece. Paris is a bustling city with a diverse population and a rich cultural heritage. It is the largest city in France and a major economic and political center. The city is known for its fashion, food, and wine industries, and is home to many famous landmarks and attractions. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations and tasks. This could lead to more efficient and effective AI systems that can perform tasks that are currently beyond the capabilities of humans.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability. This
    


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
    Generated text:  ___________. I am a/an ___________. I have a/an ___________ degree and am currently working as a/an ___________. I have been with the company for ___________ years. I bring a/an ___________ of ___________ to work, including ___________.
    
    Please describe a situation where you had to make a decision and explain your reasoning. When was this decision made, and what was the outcome? This decision was made during ___________ because ___________.
    
    Please share a current project or task that you are working on with a specific deadline. What is the current status of the project, and what
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The capital of France is Paris. It is the largest city in the European Union and one of the oldest continuously inhabited cities in the world. It is also a major cultural and economic center, known for its rich history, iconic landmarks, and diverse cuisine. Paris is home to numerous art museums, music venues, and theaters, as well as the Eiffel Tower and the Louvre Museum. The city is also known for its fashion and luxury goods, as well as its passionate, multicultural population. Paris is a major tourist destination, attracting millions of visitors each year from around the world. It is a global city with a rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting, with vast possibilities for progress and change. Here are some possible future trends in AI:
    
    1. Greater integration of AI into everyday life: AI is becoming more integrated into our daily lives, from our smartphones and IoT devices to self-driving cars. This integration will allow people to live smarter and more efficient lives, using AI-powered tools to manage their health, finances, and daily routines.
    
    2. Enhanced personalization and interaction: AI will continue to enhance personalization and interaction with users, improving the accuracy and efficiency of recommendations. This will allow for more personalized experiences, such as personalized healthcare recommendations, tailored e-commerce shopping experiences, and


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

     self

    -pro

    claimed

     writer

    .

     I

     write

     fantasy

     and

     science

     fiction

     stories

    ,

     and

     I

     enjoy

     exploring

     different

     worlds

     and

     cultures

    .

     I

    'm

     always

     on

     the

     lookout

     for

     fresh

     ideas

     and

     new

     perspectives

     on

     the

     world

    .

     If

     you

    're

     interested

     in

     sharing

     your

     work

     with

     the

     world

    ,

     don

    't

     hesitate

     to

     reach

     out

    .

     Let

    's

     connect

    .

     [

    Name

    ]

     [

    About

     them

    ,

     including

     a

     short

     bio

    ,

     a

     strong

     sense

     of

     humor

    ,

     and

     any

     hobbies

     or

     interests

     they

     have

    ].
    


    Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

    'm

     a

     self

    -pro

    claimed

     writer

    .

     I

     write

     fantasy

     and

     science

     fiction

     stories

    ,

     and

     I

     enjoy

     exploring

     different

     worlds

     and

     cultures

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Key

     facts

     about

     Paris

    :


    -

     The

     city

     is

     the

     capital

     of

     France

    


    -

     It

     is

     the

     most

     populous

     city

     in

     Europe

    


    -

     It

     is

     the

     largest

     city

     in

     the

     world

     by

     land

     area

    


    -

     It

     is

     located

     in

     the

     Î

    le

    -de

    -F

    rance

     region

    


    -

     It

     has

     a

     population

     of

     over

     

    2

    .

    5

     million

     people

    


    -

     It

     is

     known

     for

     its

     historical

     and

     cultural

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     Lou

    vre

     Museum

     


    -

     It

     is

     home

     to

     many

     museums

    ,

     including

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     the

     Lou

    vre

    ,

     and

     the

     Centre

     Pom

    pid

    ou

    


    -

     The

     city

     is

     also

     known

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     number

     of

     different

     trends

     and

     developments

    ,

     including

    :
    


    1

    .

     Increased

     availability

     of

     AI

    -powered

     tools

     and

     software

    :

     AI

     is

     becoming

     more

     widely

     available

     and

     accessible

     to

     a

     wider

     range

     of

     people

    ,

     which

     will

     lead

     to

     more

     widespread

     adoption

     and

     integration

     in

     various

     industries

    .
    


    2

    .

     AI

     becomes

     more

     widely

     used

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     medical

     diagnostics

    ,

     treatment

     planning

    ,

     and

     drug

     discovery

    ,

     but

     more

     AI

     is

     likely

     to

     be

     used

     in

     healthcare

     in

     the

     future

     to

     improve

     patient

     outcomes

     and

     reduce

     costs

    .
    


    3

    .

     AI

     is

     becoming

     more

     integrated

     into

     everyday

     life

    :

     As

     AI

     becomes

     more

     widely

     available

    ,

     it

     is

     likely

     to

     become

     more

     integrated

     into

     everyday

     life

    ,

    



```python
llm.shutdown()
```
