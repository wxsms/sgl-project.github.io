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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.72it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.71it/s]


    2026-05-16 08:58:43,705 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 08:58:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.12it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 15.12it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 15.12it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 15.12it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.18it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.00it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  31%|███       | 18/58 [00:00<00:01, 36.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.29 GB):  31%|███       | 18/58 [00:00<00:01, 36.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 36.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  31%|███       | 18/58 [00:00<00:01, 36.07it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 36.07it/s] Capturing num tokens (num_tokens=896 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 36.07it/s]Capturing num tokens (num_tokens=896 avail_mem=72.28 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.31it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.31it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.31it/s]Capturing num tokens (num_tokens=704 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.31it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.31it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.31it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=512 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.83it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.18it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.21it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.81it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.81it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.81it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.81it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.81it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.81it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 39.53it/s]


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
    Generated text:  Reith. I study at the University of Aberdeen. I'm also an associate director of the Center for Security Studies at the University of Texas at Austin. My research is on the relationship between the United States and its enemies. My book "The Iron Curtain: A History" won the 2004 Ford Prize for the best historical book. I also wrote a widely read column for "The Atlantic" and have written for "The New York Times," "The Times Literary Supplement," and "The Washington Post." I'm also a member of the Board of Directors of the Center for Strategic and International Studies. My publications include "Big
    ===============================
    Prompt: The president of the United States is
    Generated text:  25 years older than the president of Peru. The president of Peru is half the age of the president of the United States. In 5 years, what will the president of Peru be in years?
    To determine the age of the president of Peru in 5 years, we need to follow a series of logical steps based on the information given.
    
    1. Identify the current age of the president of the United States.
       - The president of the United States is 25 years older than the president of Peru.
       - Let the current age of the president of Peru be \( P \) years.
       - Therefore, the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The table shows the ticket prices for the Paris Metro (Paris's public transportation system) in different stations.
    
    | Station | Ticket Price |
    |---------|-------------|
    | A       | 100         |
    | B       | 120         |
    | C       | 150         |
    | D       | 160         |
    | E       | 180         |
    | F       | 200         |
    
    What is the highest number of tickets that can be bought for the Paris Metro if you can buy as many tickets as you want, but each station must be visited at least
    ===============================
    Prompt: The future of AI is
    Generated text:  being shaped by the progress of quantum computing, a technology that promises to revolutionize the way we think and work. Quantum computing is the science of computers that work using quantum bits, or qubits, instead of traditional bits that are either 0 or 1. Unlike classical computers, which are based on binary code, qubits can exist in multiple states at once, allowing for exponentially larger processing power. This has the potential to solve complex problems that would be impossible to solve on classical computers, such as simulating the behavior of molecules and materials, understanding the nature of quantum entanglement, and even unlocking the secrets of the brain.
    
    


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


    Generated text:  [Name] and I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of experience. I'm a [occupation] with [number] years of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. The city is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its rich history, art, and culture, and is a popular tourist destination. It is also home to many important institutions such as the French Academy of Sciences and the French National Library. The city is known for its cuisine, including its famous Parisian cuisine, and is a popular destination for food lovers. Paris is a vibrant and dynamic city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the potential trends that are likely to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient monitoring.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve fraud detection, risk management, and investment
    


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
    Generated text:  Jane, and I'm a 35-year-old software engineer. I'm passionate about coding and designing, and I love working with a team of talented individuals. I'm always looking for ways to improve my skills and keep myself up to date with the latest trends in the industry. I'm also a proud member of the Open Source community and enjoy sharing my knowledge with others. Thank you for asking! What's your favorite hobby? As an AI language model, I don't have emotions or hobbies, but I can tell you that programming is a fun hobby for many people! What's your favorite hobby? As an AI language model,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Explanation for an 8th grader: Paris is a big city where lots of people live and work. It's like a giant playground with many tall buildings and beautiful flowers in the parks. Paris has lots of yummy food like burgers and pizza. It's a special place for people to visit and learn about the country. Paris is called "the city of light" because it's bright and colorful. Isn't it fun to visit Paris? 
    
    Explanation for an 11th grader: Paris is the capital city of France. It's like a big, strong teacher who makes sure everything is running smoothly in the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be very diverse and constantly evolving, driven by a number of different trends and technologies.
    
    One of the most significant trends is the rise of automation and AI-driven automation. As technology continues to advance, we can expect to see more AI-driven automation of routine tasks, such as stock trading and customer service. This will lead to greater efficiency and productivity, but may also result in job displacement for some workers.
    
    Another trend is the rise of cognitive and machine learning. AI is increasingly being used to help people solve complex problems and make decisions. This is leading to a growing recognition of the value of AI in healthcare, finance, and other fields


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

    ,

     and

     I

     am

     a

     friendly

     and

     friendly

     kind

     of

     girl

    .

     I

     love

     to

     cook

     and

     am

     always

     looking

     for

     new

     recipes

     to

     try

     out

    .

     I

     also

     love

     to

     listen

     to

     music

    ,

     whether

     it

    's

     classical

     or

     rock

    .

     I

     enjoy

     spending

     time

     with

     my

     family

     and

     friends

    ,

     and

     I

     like

     to

     laugh

     and

     have

     fun

    .

     I

    'm

     always

     up

     for

     a

     challenge

    ,

     so

     I'm

     looking

     forward

     to

     starting

     a

     new

     recipe

     challenge

     with

     you

    .

     As

     I

     write

     this

    ,

     I

    'm

     getting

     excited

     to

     tell

     you

     about

     what

     recipes

     I

    've

     tried

     lately

     and

     how

     great

     they

     are

    !

     I

    'm

     currently

     working

     on

     my

     own

     cookbook

    ,

     and

     I

    'm

     sure

     you

    'll

     be

     interested

     in

     hearing

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     French

     capital

    ,

     Paris

    ,

     is

     a

     city

     located

     on

     the

     river

     Se

    ine

    ,

     on

     the

     Î

    le

     de

     France

    ,

     in

     the

     south

     of

     France

    ,

     with

     a

     population

     of

     around

     

    1

    .

    1

     million

     people

    .

     Paris

     is

     the

     cultural

     and

     intellectual

     center

     of

     France

     and

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    ,

     with

     its

     landmarks

    ,

     museums

    ,

     cafes

    ,

     and

     theaters

    .

     It

     is

     the

     birth

    place

     of

     many

     famous

     French

     artists

     and

     writers

    ,

     such

     as

     Victor

     Hugo

    ,

     Cam

    ille

     P

    iss

    arro

    ,

     and

     Ernest

     Hem

    ing

    way

    .

     Paris

     is

     known

     for

     its

     beautiful

     architecture

    ,

     food

    ,

     and

     fashion

    ,

     as

     well

     as

     its

     rich

     history

     and

     cultural

     heritage

    .

     As

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     undoubtedly

     going

     to

     be

     highly

     complex

     and

     varied

    ,

     with

     many

     new

     technologies

    ,

     trends

    ,

     and

     possibilities

     emerging

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     integration

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     there

     will

     likely

     be

     a

     greater

     integration

     of

     AI

     into

     various

     industries

     and

     applications

    ,

     from

     healthcare

     and

     finance

     to

     transportation

     and

     manufacturing

    .
    


    2

    .

     Enhanced

     privacy

     and

     data

     protection

    :

     As

     more

     data

     is

     collected

     and

     analyzed

     by

     AI

    ,

     there

     will

     likely

     be

     increased

     pressure

     to

     ensure

     that

     the

     data

     is

     protected

     and

     that

     it

     is

     used

     eth

    ically

     and

     responsibly

    .
    


    3

    .

     Personal

    ization

     and

     adapt

    ability

    :

     AI

     will

     be

     used

     to

     create

     more

     personalized

     and

     adaptable

     systems

    ,

     as

     well

    



```python
llm.shutdown()
```
