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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.25it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.24it/s]


    2026-05-08 10:03:31,095 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 10:03:31] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.31it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.32it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.32it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.32it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.32it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.32it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.32it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.32it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.32it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.32it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.32it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.32it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.50it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.50it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.50it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.59it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.59it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.59it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.59it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.59it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.59it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.59it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.59it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.89 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.86 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.86 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.86 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.85 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.85 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=69.85 GB):   9%|▊         | 5/58 [00:00<00:02, 21.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.83 GB):   9%|▊         | 5/58 [00:00<00:02, 21.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.82 GB):   9%|▊         | 5/58 [00:00<00:02, 21.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.82 GB):   9%|▊         | 5/58 [00:00<00:02, 21.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.82 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.82 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.17it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=69.81 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.81 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.81 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.80 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.80 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.80 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.79 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.79 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.79 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.14it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=69.79 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 23.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 23.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 23.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 23.49it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 23.49it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 23.49it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.04it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.04it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.04it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.04it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.04it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.04it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.93it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.28it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.28it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.28it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.28it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.28it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.28it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.07it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.07it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.84it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.84it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.84it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.84it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.84it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.84it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 32.50it/s]


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
    Generated text:  Anna and I am 16 years old. I am a senior in high school. I like reading and watching TV. I like to work with my friends. I am a big fan of Disney movies. My favorite actor is Frozen, and my favorite song is "Frozen". What else would you like to know about me? Please write a 100 word summary of me. Anna is a 16-year-old senior in high school. She enjoys reading, watching TV, working with friends, and being a big fan of Disney movies. Her favorite actor is Frozen and her favorite song is "Frozen". She is also a
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many foreign relations advisors to have. The president can either hire 5 people or have 7 people. The president wants to spend $100,000 and has a budget constraint of 8 people. If the president has 7 people in total, what is the maximum price the president can pay for a foreign relations advisor? Let's assume the price of a foreign relations advisor is $1,000.
    The president can hire 5 people or have 7 people, so he can choose either option.
    If he hires 5 people, the total cost would be $1,00
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. New York C. London D. Tokyo
    Answer:
    
    A
    
    Which of the following items are considered a commercial entity?
    A. A bookstore
    B. An office supply store
    C. A supermarket
    D. A hospital
    Answer:
    
    A
    
    Based on the article, which of the following statements is true?
    A. The air in the study room is warmer than the air in the living room.
    B. The study room is smaller than the living room.
    C. The air in the study room is more humid than the air in the living room.
    D. The air in the study room is
    ===============================
    Prompt: The future of AI is
    Generated text:  already here and is already changing the world around us. What’s next? You may be asking yourself – What’s the future of AI? It’s a big question, but I think we can all agree that it’s going to be a big and exciting topic.
    The field of AI is constantly evolving, and the future of AI is currently in the hands of AI researchers and developers. In this article, we will explore some of the biggest trends and developments in AI.
    Artificial intelligence (AI) has come a long way since it was first developed in the 1950s. From simple decision-making to complex problem-solving,


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


    Generated text:  [Name] and I am a [occupation] with [number of years] years of experience in [field]. I am passionate about [reason for interest] and I am always looking for ways to [action or goal]. I am a [type of person] and I am always ready to [action or goal]. I am [character trait] and I am always [action or goal]. I am [character trait] and I am always [action or goal]. I am [character trait] and I am always [action or goal]. I am [character trait] and I am always [action or goal]. I am [character
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its annual festivals and events, including the Eiffel Tower Parade and the World Cup of Lights. The city is a major transportation hub and a popular tourist destination, attracting millions of visitors each year. Paris is a cultural and intellectual center of the world and a major economic power. It is a city that has played
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI-powered healthcare: AI is already being used in healthcare to diagnose and treat diseases, and it has the potential to revolutionize the field. AI-powered healthcare systems will be able to analyze large amounts of data to
    


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
    Generated text:  [Name] and I'm a [Age] year old [occupation/character] who has lived in [location for the last [number] years] for [number] years. I have always been an [occupation or hobby], but never really achieved my full potential. Now that I'm older, I've started to believe that I should have [realistic goal], but I'm struggling to get there. I'm currently [location for the last [number] years]. I feel like I've accomplished so much, but I want to make more of a difference. What would you like to do next? Ask me something specific,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Question: What is the capital of France? The capital of France is Paris. 
    
    (A) Switzerland
    (B) Finland
    (C) Canada
    (D) Australia
    (E) Netherlands
    
    The capital of France is Paris. The correct answer is (A) Switzerland. Switzerland is not the capital of France. The capital of France is Paris. 
    
    Note: Switzerland is located on the Swiss Alps, which is not directly related to the capital of France. The French capital is Paris, known for its historic architecture, museums, and many world-renowned landmarks. The other options (Finland, Canada, and Australia) are not
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and unpredictable. However, based on current trends and projections, here are some possible future trends in AI:
    
    1. Increased AI ethics and responsibility: AI systems are increasingly being used in areas like healthcare, finance, and law enforcement, but there is growing concern about the ethical implications of AI. In the future, AI systems may be more accountable for their decisions and may have greater responsibility for their actions.
    
    2. Advancements in AI for healthcare: AI is being used to improve healthcare outcomes by improving diagnostics, personalized treatment plans, and drug discovery. AI may also be used to enhance patient care and reduce costs.
    
    3. AI-driven


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

     am

     a

     [

    role

    ]

     in

     this

     story

    .

     I

     am

     here

     to

     bring

     you

     a

     new

     perspective

     to

     the

     story

    .

     I

     aim

     to

     be

     a

     friend

     to

     [

    role

    ],

     someone

     who

     listens

     and

     respects

     [

    role

    ],

     and

     who

     can

     provide

     me

     with

     the

     information

     I

     need

     to

     be

     a

     better

     writer

    .

     How

     can

     I

     be

     of

     help

    ?

     [

    Name

    ]

    ...

     [

    Add

     more

     details

     about

     your

     role

     and

     personality

    ,

     if

     any

    ]


    Remember

    ,

     your

     purpose

     is

     to

     help

    ,

     not

     to

     steal

    .

     I

    'm

     [

    role

    ],

     your

     loyal

     friend

     and

     fellow

     writer

    .

     I

    'm

     always

     here

     to

     listen

    ,

     to

     understand

    ,

     and

     to

     help

     you

     to

     create

     the

     best

     stories

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ré

    pub

    lique

     Un

    ifi

    ée

    ."
    


    While

     Paris

     is

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

     it

     is

     also

     a

     major

     financial

     and

     cultural

     center

     of

     the

     country

    .

     The

     city

     is

     home

     to

     many

     of

     the

     world

    's

     top

     banks

    ,

     fashion

     designers

    ,

     and

     other

     influential

     institutions

    ,

     and

     has

     a

     long

     and

     stor

    ied

     history

     dating

     back

     to

     the

     Roman

     Empire

    .

     Paris

     has

     been

     a

     major

     transportation

     hub

     since

     its

     founding

     as

     a

     Roman

     colony

     in

     the

     

    1

    st

     century

     BC

    ,

     and

     continues

     to

     be

     a

     major

     economic

     and

     cultural

     center

     today

    .

     The

     city

     is

     home

     to

     a

     vast

     array

     of

     museums

    ,

     galleries

    ,

     and

     landmarks

    ,

     including

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     a

     combination

     of

     both

     positive

     and

     negative

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

     collaboration

     between

     humans

     and

     machines

    :

     As

     AI

     becomes

     more

     advanced

    ,

     we

     may

     see

     more

     collaboration

     between

     humans

     and

     machines

    .

     This

     could

     lead

     to

     increased

     efficiency

    ,

     reduced

     errors

    ,

     and

     improved

     decision

    -making

    .
    


    2

    .

     Personal

    ization

    :

     AI

     is

     already

     becoming

     more

     personal

    ,

     as

     machines

     can

     learn

     from

     users

    '

     behavior

     and

     preferences

     to

     provide

     personalized

     recommendations

     and

     experiences

    .
    


    3

    .

     Autonomous

     systems

    :

     Autonomous

     systems

    ,

     such

     as

     self

    -driving

     cars

    ,

     may

     become

     more

     common

     in

     the

     future

     as

     AI

     becomes

     more

     advanced

    .

     These

     systems

     may

     be

     able

     to

     react

     to

     unexpected

     situations

     and

     make

    



```python
llm.shutdown()
```
