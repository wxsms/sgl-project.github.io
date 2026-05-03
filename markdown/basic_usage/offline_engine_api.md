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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.71it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.71it/s]


    2026-05-03 13:20:58,705 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-03 13:20:58] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.55it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.55it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.55it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.46it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.42it/s]

    Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.42it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.42it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.46it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 39.44it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 39.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.10it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.87it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.59it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.59it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.59it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 35.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 35.68it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 35.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 35.68it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 35.68it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  31%|███       | 18/58 [00:00<00:01, 35.68it/s]Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.94it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.05it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.05it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.05it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.05it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.05it/s]

    Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.88it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.88it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  71%|███████   | 41/58 [00:01<00:00, 24.95it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  71%|███████   | 41/58 [00:01<00:00, 24.95it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  71%|███████   | 41/58 [00:01<00:00, 24.95it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  71%|███████   | 41/58 [00:01<00:00, 24.95it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 20.27it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 20.27it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 20.27it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 20.27it/s] Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 22.00it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 22.00it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 22.00it/s]

    Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 22.00it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 22.00it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  81%|████████  | 47/58 [00:01<00:00, 22.00it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 27.49it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 27.49it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 27.49it/s]Capturing num tokens (num_tokens=16 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 27.49it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 27.49it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 27.49it/s] Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 32.12it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  98%|█████████▊| 57/58 [00:01<00:00, 32.12it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 30.07it/s]


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
    Generated text:  Kaitlyn, and I am a 5th-grade student at the Cleveland School. I have been learning about government and the environment for the past year. I have also been learning about coral reefs, and I love learning about marine life and marine ecosystems. I have been writing a poem about coral reefs and their importance to marine life.
    Can you please summarize the poem I wrote and what it explores about coral reefs?
    I'm sorry, but as an AI language model, I don't have the ability to create a poem or have the capability to see or access images. However, I can help you with any questions you may have about
    ===============================
    Prompt: The president of the United States is
    Generated text:  a type of political leader. He is the head of the government and the leader of the country. President is the highest position of the government in the United States. The term of the president is 4 years. The president of the United States usually is re-elected for a second term.
    
    The President has several jobs to do, including:
    
    1. To have the first priority of making the policies for the country. The president of the United States is responsible to make the policies and make decisions for the country.
    
    2. To run the country's government and the executive branch. The president is the head of the government, and the president has
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    
    B) London
    
    C) Rome
    
    D) Berlin
    
    1. **Understanding the Problem:**
       We are given a list of options representing capital cities in France. Our task is to identify which capital city from this list matches the statement "the capital of France is Paris."
    
    2. **Analyzing the Options:**
       - **Paris:** The capital city of France is Paris, which is listed in the given options.
       - **London:** London is the capital city of the United Kingdom, not France. This does not match the statement.
       - **Rome:** Rome is the capital city of
    ===============================
    Prompt: The future of AI is
    Generated text:  not clear; we will have to wait and see how the market evolves. But there is no denying that the rise of AI has had a major impact on the world, and it has created an environment in which the companies that are most prepared to adapt can thrive.
    One of the key areas in which AI will have a significant impact is the healthcare industry. With AI, healthcare professionals can be trained to analyze data and provide more accurate diagnoses and treatments. For example, AI-powered diagnostic tools can help doctors identify patterns and make more accurate predictions about patient outcomes. This can help improve patient care and reduce healthcare costs.
    AI can also be used to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter, where many famous French artists and writers live and work. Paris is a bustling city with a rich history and culture, and it is a popular tourist destination. The city is also known for its fashion industry, with many famous fashion designers and boutiques located in the city. Overall, Paris is a city of art, culture, and history that is a must-visit for anyone interested in France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation: AI is likely to become more prevalent in many industries, including manufacturing, transportation, and healthcare. Automation will likely lead to increased efficiency and productivity, but it will also lead to the loss of jobs for some workers.
    
    2. AI ethics and privacy: As AI becomes more advanced, there will likely be increased scrutiny of its use and potential misuse. There will be a push for greater transparency and accountability in the development and use
    


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
    Generated text:  [insert name] and I'm [insert age]. I'm a [insert occupation] with a [insert education] degree in [insert field of study]. I'm passionate about [insert what you enjoy/what you love/what you enjoy doing], and I'm always on the lookout for opportunities to [insert what you are looking for/what you are up to]. I'm very [insert trait/characteristic/positive trait] and I strive to be [insert how you can show your positive traits/characteristic]. I'm a [insert what you're known for/what you're good at/what you're most famous
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    What is the capital of France and where is it located? The capital of France is Paris. It is located in the central region of the country, in the south-central part of France. Paris is the largest city in France by population, with a population of 1, 400, 000 according to the 2019 population count. Paris is known as the "city of love," and many tourists come to the city to see the Eiffel Tower, Notre Dame Cathedral, and other iconic landmarks. It is also home to the Eiffel Tower, Louvre Museum, Opera House
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of possibilities and unknowns, and it is difficult to predict exactly where it will lead. However, there are some trends that are likely to continue or evolve in the coming years:
    
    1. Increase in AI intelligence: AI is becoming more intelligent and capable of performing complex tasks, such as natural language processing, decision-making, and image recognition. In the future, we may see AI systems become even more sophisticated and capable of understanding human emotions and desires.
    
    2. AI integration with human decision-making: With the increasing amount of data being generated, it is becoming increasingly difficult for humans to make decisions. AI systems may become more integrated with human


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

    Your

     Name

    ]

     and

     I

     am

     a

     [

    Your

     Profession

     or

     Occupation

    ].

     I

     have

     been

     in

     the

     [

    Your

     Profession

     or

     Occupation

    ]

     industry

     for

     [

    X

     years

     or

     more

    ],

     and

     I

     am

     proud

     to

     call

     myself

     a

     [

    Your

     Profession

     or

     Occupation

    ]

    !

     I

    'm

     a

     [

    Your

     Profession

     or

     Occupation

    ]

     who

     values

     [

    Your

     Qual

    ifications

     or

     Skills

    ]

     above

     all

    .

     And

     when

     it

     comes

     to

     my

     passion

    ,

     I

     believe

     in

     [

    Your

     Passion

     Statement

    ].

     I

    'm

     always

     ready

     to

     help

     anyone

     who

     needs

     it

    ,

     and

     I

    'm

     always

     looking

     for

     the

     next

     challenge

    .

     I

    'm

     always

     learning

     and

     growing

    ,

     and

     I

    'm

     excited

     to

     see

     where

     this

     journey

     takes

     me

    !

     Let

    's

     connect

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Its

     name

     comes

     from

     the

     Latin

     term

     meaning

     "

    city

     of

     the

     living

    ."

     Paris

     is

     a

     historic

     city

     known

     for

     its

     iconic

     landmarks

     like

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     It

    's

     the

     largest

     city

     in

     France

    ,

     with

     a

     population

     of

     over

     

    2

    .

    7

     million

     people

    .

     Paris

     is

     also

     renowned

     for

     its

     rich

     history

     and

     cultural

     attractions

    .

     It

     is

     the

     economic

     and

     political

     heart

     of

     France

     and

     the

     capital

     of

     the

     European

     Union

    ,

     hosting

     the

     European

     Parliament

    .

     The

     city

     is

     a

     major

     center

     of

     science

    ,

     art

    ,

     music

    ,

     fashion

    ,

     and

     more

    ,

     attracting

     tourists

     and

     residents

     from

     around

     the

     world

    .

     Paris

     is

     also

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     has

     already

     started

     to

     reveal

     many

     interesting

     possibilities

    .

     Some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     has

     the

     potential

     to

     revolution

    ize

     the

     way

     we

     diagnose

     and

     treat

     diseases

    .

     AI

    -powered

     diagnostic

     tools

     could

     help

     doctors

     make

     more

     accurate

     diagnoses

    ,

     identify

     patients

     at

     higher

     risk

     of

     developing

     certain

     diseases

    ,

     and

     personalize

     treatment

     plans

    .
    


    2

    .

     Automation

     of

     jobs

    :

     While

     AI

     has

     the

     potential

     to

     automate

     many

     tasks

    ,

     it

     is

     also

     likely

     to

     dis

    place

     some

     jobs

    .

     This

     could

     lead

     to

     increased

     job

     insecurity

     and

     economic

     inequality

    .
    


    3

    .

     AI

    -powered

     education

    :

     AI

     could

     be

     used

     to

     personalize

     learning

     experiences

     for

     students

    ,

     providing

     individual

    



```python
llm.shutdown()
```
