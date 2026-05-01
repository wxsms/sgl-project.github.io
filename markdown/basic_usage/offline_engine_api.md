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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.23it/s]


    2026-05-01 21:01:23,840 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-01 21:01:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:36,  4.85s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.03it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.03it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.09it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.09it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.09it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.09it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.09it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.09it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.09it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.09it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.09it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.09it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.46it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 22.46it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 22.46it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 31.09it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 31.09it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 31.09it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 31.09it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 31.09it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 31.09it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 31.09it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 31.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.61 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.61 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.61 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.55 GB):   9%|▊         | 5/58 [00:00<00:02, 21.11it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.54 GB):   9%|▊         | 5/58 [00:00<00:02, 21.11it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=56.54 GB):   9%|▊         | 5/58 [00:00<00:02, 21.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.54 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.53 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.39it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=56.53 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.39it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.53 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.16it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.53 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.52 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.16it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=56.52 GB):  21%|██        | 12/58 [00:00<00:03, 11.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.52 GB):  21%|██        | 12/58 [00:00<00:03, 11.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.52 GB):  21%|██        | 12/58 [00:01<00:03, 11.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.52 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.51 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.51 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.30it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=56.51 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.50 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.50 GB):  31%|███       | 18/58 [00:01<00:02, 17.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.50 GB):  31%|███       | 18/58 [00:01<00:02, 17.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.50 GB):  31%|███       | 18/58 [00:01<00:02, 17.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.48 GB):  31%|███       | 18/58 [00:01<00:02, 17.67it/s]Capturing num tokens (num_tokens=960 avail_mem=56.49 GB):  31%|███       | 18/58 [00:01<00:02, 17.67it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=56.49 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.33it/s]Capturing num tokens (num_tokens=896 avail_mem=74.14 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.33it/s]Capturing num tokens (num_tokens=832 avail_mem=74.14 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.33it/s]Capturing num tokens (num_tokens=768 avail_mem=74.13 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.33it/s]Capturing num tokens (num_tokens=704 avail_mem=74.13 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.33it/s]Capturing num tokens (num_tokens=640 avail_mem=74.13 GB):  38%|███▊      | 22/58 [00:01<00:01, 18.33it/s]Capturing num tokens (num_tokens=640 avail_mem=74.13 GB):  47%|████▋     | 27/58 [00:01<00:01, 24.12it/s]Capturing num tokens (num_tokens=576 avail_mem=74.12 GB):  47%|████▋     | 27/58 [00:01<00:01, 24.12it/s]Capturing num tokens (num_tokens=512 avail_mem=74.11 GB):  47%|████▋     | 27/58 [00:01<00:01, 24.12it/s]Capturing num tokens (num_tokens=480 avail_mem=74.13 GB):  47%|████▋     | 27/58 [00:01<00:01, 24.12it/s]Capturing num tokens (num_tokens=448 avail_mem=74.13 GB):  47%|████▋     | 27/58 [00:01<00:01, 24.12it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.12 GB):  47%|████▋     | 27/58 [00:01<00:01, 24.12it/s]Capturing num tokens (num_tokens=416 avail_mem=74.12 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.36it/s]Capturing num tokens (num_tokens=384 avail_mem=74.12 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.36it/s]Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.36it/s]Capturing num tokens (num_tokens=320 avail_mem=74.11 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.36it/s]Capturing num tokens (num_tokens=288 avail_mem=74.11 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.36it/s]Capturing num tokens (num_tokens=256 avail_mem=74.10 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.36it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.36it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=208 avail_mem=74.09 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.11it/s]

    Capturing num tokens (num_tokens=176 avail_mem=74.05 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.50it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.50it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.50it/s]Capturing num tokens (num_tokens=112 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.50it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.50it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.50it/s]Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.54it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.54it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.54it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.54it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  83%|████████▎ | 48/58 [00:02<00:00, 40.54it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  83%|████████▎ | 48/58 [00:02<00:00, 40.54it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.66it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.66it/s]Capturing num tokens (num_tokens=16 avail_mem=74.01 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.66it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.66it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.66it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.66it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:02<00:00, 26.93it/s]


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
    Generated text:  Tony and I'm a digital marketer. I have a small business that sells handmade books, writing materials, and other items for creative inspiration. I'm in the process of launching a new e-commerce site. What are some strategies I can use to ensure my site is optimized for search engines?
    
    To do this, please provide me with a detailed plan, including at least three steps, that I can take to enhance my site's SEO. Additionally, please provide me with a comparison of my current SEO practices with the best practices to make the most of my website. 
    
    To make my optimization efforts more effective, I would like to incorporate a user
    ===============================
    Prompt: The president of the United States is
    Generated text:  in New York and the president of the European Union is in France. Which two countries are both located in Europe?
    A) The United States and the United Kingdom
    B) The United States and France
    C) The United Kingdom and France
    D) The United Kingdom and Germany
    E) Germany and France
    To determine which two countries are both located in Europe, let's analyze the information given:
    
    1. The president of the United States is in New York.
    2. The president of the European Union is in France.
    
    Since the United States is located in North America, which is not Europe, and the European Union is located in
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and the capital of Brazil is Brasilia. Which of the following statements about Paris and Brasilia is true? 
    A: Paris is in the Western Hemisphere, Brazil is in the Southern Hemisphere
    B: Paris is the largest city in the world, while Brasilia is the smallest city in the world
    C: Paris is known as the 'City of Light', while Brasilia is known as the 'City of Bricks'
    D: Paris is located on the Atlantic coast, while Brasilia is located in the interior of the continent
    
    To determine the correct statement about Paris and Brasilia, we need to analyze each option carefully
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but you can't predict the future exactly. What does that imply for the people who work in AI?
    
    As the AI revolution continues to grow and improve at an exponential rate, it's essential for employees to understand that there is a lot of room for growth and advancement. This doesn't mean that you have to let the technology dictate your career path or make it dependent on AI. Instead, you need to find a balance between working within the limits of the technology and continuing to learn and improve.
    
    In order to do this, it's important to stay up to date with the latest developments in AI and be open to new technologies and


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] with a passion for [Interest]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Personality] person who is always [Adjective]. I'm [Job Title] at [Company Name], and I'm excited to be here. I'm looking forward to [Future Goal] and I'm always looking for ways to make a positive impact in the world. I'm a [Personality] person who is always [Adjective]. I'm [Job Title] at [Company Name],
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower, and its rich history dating back to the Middle Ages. It is also home to the Louvre Museum, the most famous art museum in the world, and the Notre-Dame Cathedral, a stunning Gothic structure. Paris is a vibrant and diverse city with a rich cultural scene, and it is a popular tourist destination. The city is also known for its fashion industry, with many famous fashion houses and designers operating in the area. Overall, Paris is a city of contrasts and excitement, and it is a must-visit destination for anyone interested in French culture and history.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical AI: As more and more AI systems are being developed, there will be a greater emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and fairness. As AI systems become more complex and sophisticated, it is likely that they will need to be designed with these ethical considerations in mind.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient
    


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
    Generated text:  [Name]. I'm a [Career/Job Title] at [Company]. I'm really excited to meet you and learn more about you. How can I assist you today? [Name]: Hi there! Nice to meet you. I'm [Name] and I work at [Company]. I really enjoy my job here and I'm always looking for ways to improve my skills and knowledge. How can I assist you today? [Name]: Oh, that's very kind of you to say! I'm here to learn and grow and I'm really interested in [Company's name]. I think we have a lot in common. Thanks
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city and most populous metropolitan area in the European Union, and one of the most visited cities in the world. The city is known for its stunning architecture, rich cultural heritage, and vibrant street life. It is also one of the most important financial centers in the world, and hosts several major international events annually. Despite its size, Paris is known for its intellectual and artistic life, and is a city of contrasts, with elements of Gothic, modern, and classical architecture. The city has a rich history dating back to the Roman Empire and the Crusades, and continues to be an important cultural and economic hub in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a combination of technological advancements, changing societal needs, and evolving human values. Here are some possible trends in the AI landscape:
    
    1. Increased focus on ethical AI: As concerns about AI's impact on society grow, there is likely to be increased focus on ethical AI. This could involve developing more transparent and accountable AI systems, as well as guidelines for how AI should be used in various applications.
    
    2. Integration with other technologies: AI is already integrated into a wide range of products and services, from smart home devices to facial recognition systems. As more of these technologies are integrated into everyday life, there is likely to


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

    ]

     and

     I

     am

     a

     [

    job

     title

    ]

     at

     [

    Company

     Name

    ].

     I

     have

     a

     passion

     for

     [

    mention

     something

     specific

     that

     defines

     you

    ].

     What

     kind

     of

     experience

     do

     you

     have

     that

     makes

     you

     a

     good

     fit

     for

     this

     job

     role

    ?


    [

    Name

    ]

     [

    Job

     Title

    ]

     at

     [

    Company

     Name

    ]

     brings

     a

     wealth

     of

     knowledge

    ,

     experience

    ,

     and

     a

     zest

     for

     life

     that

     make

     me

     a

     great

     fit

     for

     this

     position

    .

     I

     am

     always

     seeking

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     always

     looking

     for

     opportunities

     to

     contribute

     to

     my

     team

     and

     make

     a

     positive

     impact

     on

     the

     company

    .


    I

     have

     a

     passion

     for

     [

    mention

     something

     specific

     that

     defines

     you

    ]

     and

     I

     am

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     statement

     is

     accurate

    ,

     and

     Paris

     is

     indeed

     the

     capital

     city

     of

     France

    .

     It

     is

     the

     largest

     city

     in

     the

     European

     Union

     and

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     culture

    .

     The

     city

     is

     located

     in

     the

     southeastern

     part

     of

     France

    ,

     on

     the

     River

     Se

    ine

     and

     surrounded

     by

     hills

    .

     It

     is

     home

     to

     many

     famous

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     art

     scene

    ,

     and

     its

     role

     in

     the

     French

     revolution

    .

     Its

     population

     is

     around

     

    2

    .

    7

     million

     as

     of

     the

     

    2

    0

    2

    1

     census

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     constantly

     evolving

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     Transparency

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     is

     a

     higher

     demand

     for

     transparency

    .

     This

     could

     lead

     to

     increased

     scrutiny

     of

     AI

     algorithms

     and

     their

     decision

    -making

     processes

    .
    


    2

    .

     Development

     of

     AI

     Ethics

    :

     As

     AI

     becomes

     more

     advanced

    ,

     there

     will

     be

     a

     growing

     concern

     about

     its

     ethical

     implications

    .

     This

     could

     lead

     to

     a

     push

     for

     developing

     new

     ethical

     guidelines

     for

     AI

    .
    


    3

    .

     AI

     Integration

     in

     Healthcare

    :

     AI

     has

     the

     potential

     to

     revolution

    ize

     healthcare

     by

     improving

     patient

     outcomes

     and

     cost

     savings

    .

     However

    ,

     there

     is

     a

     need

     for

     better

     data

     privacy

    



```python
llm.shutdown()
```
