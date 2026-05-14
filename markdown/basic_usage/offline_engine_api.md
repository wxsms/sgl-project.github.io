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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.83it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.82it/s]


    2026-05-14 23:14:58,462 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 23:14:58] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.01it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  9.01it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.14it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.14it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.14it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.14it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.14it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.14it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.14it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.14it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.14it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.14it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 22.43it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 22.43it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 22.43it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 22.43it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 22.43it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 22.43it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 22.43it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 22.43it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 22.43it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 22.43it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:04<00:00, 22.43it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 31.69it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 31.69it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 31.69it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 31.69it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 31.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   9%|▊         | 5/58 [00:00<00:02, 21.61it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.12 GB):   9%|▊         | 5/58 [00:00<00:02, 21.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.10 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.10 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.25it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.05 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.05 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.04 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.04 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.04 GB):  31%|███       | 18/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.04 GB):  31%|███       | 18/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.04 GB):  31%|███       | 18/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.02 GB):  31%|███       | 18/58 [00:00<00:01, 34.57it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.03 GB):  31%|███       | 18/58 [00:00<00:01, 34.57it/s] Capturing num tokens (num_tokens=896 avail_mem=72.03 GB):  31%|███       | 18/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=896 avail_mem=72.03 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=832 avail_mem=72.02 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=768 avail_mem=72.02 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=704 avail_mem=72.02 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=640 avail_mem=72.01 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=576 avail_mem=72.01 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=576 avail_mem=72.01 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.42it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.42it/s]Capturing num tokens (num_tokens=480 avail_mem=72.01 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.42it/s]Capturing num tokens (num_tokens=448 avail_mem=72.01 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.42it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.42it/s]Capturing num tokens (num_tokens=384 avail_mem=72.01 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.42it/s]Capturing num tokens (num_tokens=384 avail_mem=72.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=352 avail_mem=72.00 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=320 avail_mem=72.00 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=288 avail_mem=71.99 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=240 avail_mem=71.99 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.36it/s]Capturing num tokens (num_tokens=240 avail_mem=71.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.89it/s]Capturing num tokens (num_tokens=224 avail_mem=71.98 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.89it/s]Capturing num tokens (num_tokens=208 avail_mem=71.98 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.89it/s]

    Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.89it/s]Capturing num tokens (num_tokens=176 avail_mem=71.98 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.89it/s]Capturing num tokens (num_tokens=160 avail_mem=71.98 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.89it/s]Capturing num tokens (num_tokens=160 avail_mem=71.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=144 avail_mem=71.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=128 avail_mem=71.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=112 avail_mem=71.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=96 avail_mem=71.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.78it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=71.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=80 avail_mem=71.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=64 avail_mem=71.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=48 avail_mem=71.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=32 avail_mem=71.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=28 avail_mem=71.94 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.78it/s]Capturing num tokens (num_tokens=20 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.78it/s]Capturing num tokens (num_tokens=16 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.78it/s]Capturing num tokens (num_tokens=12 avail_mem=71.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.78it/s]

    Capturing num tokens (num_tokens=8 avail_mem=71.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.78it/s] Capturing num tokens (num_tokens=4 avail_mem=71.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.78it/s]Capturing num tokens (num_tokens=4 avail_mem=71.93 GB): 100%|██████████| 58/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=4 avail_mem=71.93 GB): 100%|██████████| 58/58 [00:01<00:00, 37.61it/s]


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
    Generated text:  Bishong and I am a professional. I am going to travel and will be a tourist. Please give me some advice on how to prepare for a tour, including packing, transportation, and safety.
    
    Sure, I'd be happy to help! Here are some general tips for preparing for a tour:
    
    Packing:
    - Make sure you have a well-stocked backpack with all the necessary items for your trip
    - Bring a comfortable, low-slung bag for easy carrying
    - Make sure you have all the necessary items for your destination (e.g. camera, maps, wallets, etc.)
    - Pack light, but don't forget
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have in the country. He knows from past experiences that the number of military bases will be inversely proportional to the square root of the country's population. In 2010, the country had a population of 300,000 and the number of military bases was 5. If the country's population doubles in 2020, how many military bases will the president have then? To solve this problem, we need to understand that the number of military bases is inversely proportional to the square root of the population. This means that if the number of military
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. What is the third word in the Latin alphabet? The third word in the Latin alphabet after "Paris" is "de". 
    
    To justify this, we examine the sequence of the Latin alphabet:
    
    1. A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
    
    The sequence of words we are looking for is not immediately obvious, so let's break down the sequence:
    
    1. "Paris" is the third word in the alphabet
    ===============================
    Prompt: The future of AI is
    Generated text:  changing rapidly. So, it is important to keep on learning and stay up-to-date on the latest developments. That’s where the joint venture between the University of Sydney and the Australian National University comes into play. The two universities recently co-developed an algorithm to improve the accuracy of medical diagnosis by analyzing real-world medical images.
    The research was published in Nature, a leading scientific journal.
    It’s a significant milestone for AI, as medical diagnoses are currently done manually, which can be prone to errors. Artificial intelligence can help by analyzing images and improving diagnostic accuracy, even when it comes to diagnosing complex conditions like cancer.
    The team led by


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic center with a rich history and a diverse population of over 10 million people. It is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its fashion, art, and cuisine, and is a major center for science and technology. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to inspire and influence the world
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn and adapt to new situations more effectively. This could lead to more sophisticated and adaptive AI systems that can perform tasks that require human-like intelligence.
    
    2. Enhanced machine learning capabilities: AI systems are likely to become even more capable of learning and adapting to new situations, thanks to advances in machine learning algorithms. This could lead to more sophisticated and personalized AI systems that can perform tasks that require human-like intelligence.
    
    3. Increased use of AI in healthcare: AI
    


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
    Generated text:  [Name], and I'm a/an [Position/Job] at [Company Name]. I'm here to bring value to [Company Name] through [Specific Skills or Knowledge], and I'm excited to help you achieve your goals. How can I be of service? I look forward to [Looking Forward to] and [Engaging]. Let's create a relationship based on mutual trust and respect. Looking forward to [Next Steps]. Cheers! [Name] [Company Name] [Position/Job] [Company Name] [Company Name] [Company Name] [Company Name] [Company Name] [Company Name] [Company Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris is the capital city of France and the largest city in the European Union. It is located in the northern part of the country and is known for its museums, museums, opera house, and nightlife. Paris is also a major cultural and economic center and is home to many famous landmarks such as the Eiffel Tower and Notre-Dame Cathedral. The city is also known for its rich history and unique architecture, which is evident in its many museums, theaters, and other cultural institutions. Overall, Paris is a fascinating and vibrant city that is home to millions of residents and visitors each year.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and depends on a variety of factors, including technological advancements, shifts in business and societal demands, and regulatory changes. Here are some potential future trends in AI:
    
    1. Improved safety and security in AI applications: As AI continues to become more prevalent in various sectors, there is a growing need to ensure that it is used safely and securely. This includes developing more robust AI algorithms that can identify and mitigate potential risks, such as cyber attacks and biased data.
    
    2. AI personalization and autonomy: As AI systems become more capable, they are likely to become even more personalized and autonomous. This could lead to new applications and industries that


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

    insert

     fictional

     character

    's

     name

    ],

     and

     I

     am

     an

     AI

     assistant

     created

     by

     Anth

    ropic

    .

     I

     am

     here

     to

     help

     you

     with

     any

     questions

     you

     might

     have

    ,

     answer

     any

     questions

     you

     might

     have

    ,

     and

     even

     assist

     with

     writing

     papers

     for

     you

    .

     I

     am

     here

     to

     be

     your

     go

    -to

     resource

     for

     any

     questions

     you

     might

     have

     about

     the

     world

    .

     I

     am

     always

     ready

     to

     help

     you

     with

     anything

     that

     you

     need

    ,

     whether

     it

    's

     an

     answer

     to

     a

     question

    ,

     a

     suggestion

     for

     a

     topic

    ,

     or

     even

     a

     piece

     of

     advice

    .

     So

    ,

     if

     you

     need

     help

     with

     anything

    ,

     don

    't

     hesitate

     to

     reach

     out

     to

     me

    .

     I

     am

     here

     to

     be

     your

     go

    -to

     resource

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     in

     the

     country

     and

     the

     heart

     of

     the

     French

     Empire

    .


    Paris

     has

     a

     long

     history

     dating

     back

     to

     the

     Roman

     era

    .

     The

     city

     is

     a

     cultural

     and

     economic

     center

    ,

     with

     important

     museums

    ,

     theaters

    ,

     and

     historical

     landmarks

    .

     Its

     rich

     history

     and

     diverse

     culture

     have

     made

     it

     a

     popular

     tourist

     destination

    ,

     attracting

     visitors

     from

     around

     the

     world

    .

     Paris

     is

     also

     home

     to

     many

     prestigious

     universities

     and

     institutions

     of

     higher

     learning

    .

     It

     is

     a

     major

     transportation

     hub

    ,

     offering

     high

    -speed

     train

     services

     and

     public

     transportation

    .

     Paris

     has

     a

     strong

     economy

    ,

     with

     a

     thriving

     fashion

     industry

    ,

     music

    ,

     and

     arts

     scene

    .

     It

     is

     also

     known

     for

     its

     culinary

     scene

    ,

     with

     many

     famous

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     potential

     possibilities

     and

     exciting

     developments

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

     Human

    -A

    I

     collaboration

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     expect

     to

     see

     an

     increase

     in

     human

    -A

    I

     collaboration

    .

     This

     could

     be

     through

     the

     use

     of

     more

     advanced

     AI

     systems

     that

     can

     mimic

     human

     intelligence

     and

     decision

    -making

    ,

     or

     through

     the

     development

     of

     more

     personalized

     and

     adaptable

     AI

     that

     can

     adapt

     to

     new

     situations

     and

     needs

    .
    


    2

    .

     AI

    -driven

     healthcare

     advancements

    :

     AI

     is

     already

     being

     used

     to

     improve

     healthcare

     outcomes

    ,

     and

     we

     expect

     to

     see

     even

     more

     significant

     advancements

     in

     the

     future

    .

     AI

     can

     help

     with

     diagnostics

    ,

     treatment

     planning

    ,

     and

     patient

     monitoring

    ,

     and

     can

    



```python
llm.shutdown()
```
