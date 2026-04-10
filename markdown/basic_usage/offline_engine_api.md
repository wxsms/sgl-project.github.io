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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.64it/s]


    2026-04-10 21:00:17,124 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 21:00:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.09it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.09it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.09it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.09it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.09it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  6.09it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.09it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.09it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.09it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.34it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.34it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.34it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.34it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.34it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.34it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.34it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.34it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.91it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.91it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.91it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.91it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.91it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.91it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.91it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.88it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.70it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.70it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.70it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.70it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.70it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.70it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.70it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.68it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.24it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.60it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.16it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.52it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.52it/s] Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.52it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.52it/s]

    Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.32it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.32it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.32it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.32it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.32it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.32it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  50%|█████     | 29/58 [00:01<00:01, 24.66it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:01, 24.66it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:01, 24.66it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:01, 24.66it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:01, 24.66it/s]Capturing num tokens (num_tokens=352 avail_mem=120.26 GB):  50%|█████     | 29/58 [00:01<00:01, 24.66it/s]

    Capturing num tokens (num_tokens=352 avail_mem=120.26 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.30it/s]Capturing num tokens (num_tokens=320 avail_mem=120.26 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.30it/s]Capturing num tokens (num_tokens=288 avail_mem=119.03 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.30it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.30it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.30it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.77it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.77it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.77it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.77it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.77it/s]

    Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.77it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.50it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.50it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.50it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.50it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.50it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.50it/s]Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.49it/s]Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.49it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.49it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.49it/s]

    Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.49it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.49it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.66it/s] Capturing num tokens (num_tokens=4 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:01<00:00, 31.63it/s]


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
    Generated text:  1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political leader, and the president is the head of the executive branch of the federal government. What can the president do? Choose the most suitable option to complete the sentence.
    
    A) Manage the economy;  
    B) Promote science;  
    C) Organize the military;  
    D) Negotiate with foreign governments;  
    E) Be a film director;  
    F) Be a comedian;  
    G) Serve as a professional athlete;  
    H) Serve as a professional athlete;  
    
    The president is a political leader, and the president is the head of the executive branch of the federal government. What can the president do?
    
    Choose
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lyon
    C. Rennes
    D. Bordeaux
    
    To determine the capital of France, we need to consider the most important cities that are internationally recognized as the capital of France. These cities are typically the largest and most significant in terms of cultural, economic, and historical importance.
    
    Let's analyze the options given:
    
    A. Paris
    B. Lyon
    C. Rennes
    D. Bordeaux
    
    Paris is the capital of France, and it is one of the most important cities in Europe. Lyon, Rennes, and Bordeaux are all smaller cities in France, but they are not recognized as the
    ===============================
    Prompt: The future of AI is
    Generated text:  here – and it’s not just about the technology itself. The field of AI is rapidly growing, evolving and reshaping industries, professions, and even our society. A growing number of organizations, businesses and researchers are investing in AI, and technology companies are increasingly asking what the future of AI looks like.
    We’ve explored the future of AI and its possible impacts on society and the economy. We believe that the development of a robust AI is essential to help create a better future. If you want to learn more about the future of AI, then we recommend reading the research papers and reports and the news updates. You can find more information about


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few key points about your background, skills, or personality]. I'm always looking for new opportunities to grow and learn, and I'm always eager to share my knowledge and experience with others. Thank you for taking the time to meet me. [Name] [Company name] [Job title] [Company website] [LinkedIn profile] [Twitter handle] [Facebook page] [Email address] [Phone number] [Website]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art, and cuisine. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. It is a popular tourist destination and a major economic center in Europe. The city is also home to many famous museums, theaters, and restaurants. Paris is a city of contrasts, with its modern architecture and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation: As AI becomes more advanced, it is likely to become more efficient and effective at performing tasks that were previously done by humans. This could lead to increased automation in industries such as manufacturing, transportation, and healthcare.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there will be a greater need for privacy and security measures to protect the data that is generated and used by AI systems
    


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
    Generated text:  ___________, and I'm a/an ___________ writer.
    Hi there! I'm a/an ___________ writer. I specialize in ___________. What kind of stories do you like to write? And where are you from? Let's make a great connection! 🌍✨
    I'm a/an ___________ writer who specializes in writing ___________. I'm excited to start this journey with you! 🚀✨
    Here's a story idea for you: Can you describe a moment that stands out to you, a setting, or a character that inspires you? 🌍✨
    Here's your chance to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city famous for its grand cathedrals, famous landmarks, and rich history. It is the largest city in Europe, with a population of over 2 million people. Paris is also known as the "city of love" and "La Ville Blanche," and it is home to many important historical sites, museums, and landmarks. Paris has been a world-renowned city for over 2000 years, and it is now a major cultural and economic center in the world. It is considered to be one of the most beautiful cities in the world, with its winding streets, charming boulevards, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, and it is expected to continue to evolve and expand in many exciting ways. Here are some possible trends in AI that are likely to shape the industry in the coming years:
    
    1. Increased focus on ethical AI: As more and more companies and individuals become concerned about the potential consequences of AI, there will be a greater emphasis on developing ethical AI. This includes ensuring that AI systems are designed to minimize bias, discrimination, and other negative effects.
    
    2. More integration with other technologies: AI is increasingly becoming integrated with other technologies, such as sensors, data analysis, and machine learning algorithms. This integration will allow for more precise and


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

    ],

     and

     I

    'm

     a

     [

    short

    ,

     neutral

     title

     for

     you

    ].

     I

     have

     been

     in

     this

     field

     for

     [

    number

    ]

     years

     now

    ,

     and

     I

    've

     always

     been

     passionate

     about

     [

    explain

     your

     field

     of

     interest

    ].

     I

     work

     with

     people

     from

     all

     walks

     of

     life

    ,

     and

     I

     always

     try

     to

     be

     open

    -minded

     and

     empath

    etic

     towards

     everyone

     I

     meet

    .

     I

     have

     a

     great

     sense

     of

     humor

    ,

     and

     I

    'm

     always

     up

     for

     a

     good

     laugh

    .

     I

    'm

     [

    short

    ,

     neutral

     title

     for

     you

    ],

     and

     I

     hope

     to

     have

     a

     great

     day

    !

     [

    End

     of

     intro

    ].

     (

    This

     is

     an

     example

     of

     a

     self

    -int

    roduction

    ,

     but

     the

     actual

     title

     and

     placeholder

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

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

     Paris

     is

     also

     a

     popular

     tourist

     destination

     known

     for

     its

     diverse

     culture

    ,

     cuisine

    ,

     and

     fashion

     industry

    .

     The

     French

     language

     is

     an

     official

     language

     of

     the

     country

    ,

     and

     it

     is

     widely

     spoken

     in

     the

     city

    .

     Paris

     is

     a

     cultural

     melting

     pot

    ,

     with

     a

     rich

     history

     and

     a

     vibrant

     social

     scene

    .

     It

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Light

    "

     and

     has

     been

     the

     seat

     of

     the

     French

     government

     and

     its

     capital

     since

     the

     

    1

    3

    th

     century

    .

     Paris

     has

     been

     a

     global

     city

     for

     over

     

    3

    0

    0

     years

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

     are

     some

     potential

     trends

     that

     could

     emerge

    :
    


    1

    .

     Increased

     automation

     and

     AI

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     likely

     to

     automate

     more

     tasks

    ,

     increasing

     efficiency

     and

     productivity

     in

     many

     industries

    .

     This

     could

     lead

     to

     job

     displacement

     and

     create

     new

     opportunities

     in

     areas

     such

     as

     data

     analysis

    ,

     cybersecurity

    ,

     and

     machine

     learning

    .
    


    2

    .

     Development

     of

     ethical

     AI

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

     need

     to

     develop

     ethical

     guidelines

     for

     its

     development

     and

     use

    .

     This

     could

     lead

     to

     increased

     regulation

     and

     scrutiny

     of

     AI

     systems

    ,

     particularly

     those

     that

     are

     used

     for

     decision

    -making

     in

     certain

     contexts

    .
    


    3

    .

     AI

    -powered

     healthcare

    :

     AI

     is

     being

     increasingly

     used

    



```python
llm.shutdown()
```
