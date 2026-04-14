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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.25it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.24it/s]


    2026-04-14 08:17:58,118 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 08:17:58] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.69it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.05it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.36it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.36it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.36it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.36it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.36it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.36it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.36it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.36it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 25.29it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 25.29it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 25.29it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 25.29it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 25.29it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 25.29it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 25.29it/s]

    Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:01, 25.29it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 38.43it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 38.43it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 38.43it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 38.43it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 38.43it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 38.43it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 38.43it/s]

    Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 38.43it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 38.43it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 46.52it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 46.52it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 46.52it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 46.52it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 46.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.62 GB):   2%|▏         | 1/58 [00:00<00:09,  5.85it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=72.59 GB):   2%|▏         | 1/58 [00:00<00:09,  5.85it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:07,  7.02it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.58 GB):   3%|▎         | 2/58 [00:00<00:07,  7.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.58 GB):   3%|▎         | 2/58 [00:00<00:07,  7.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.58 GB):   3%|▎         | 2/58 [00:00<00:07,  7.02it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.76it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.76it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.76it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.76it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.76it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.76it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.76it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.76it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.76it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.97it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.97it/s] Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.97it/s]Capturing num tokens (num_tokens=832 avail_mem=72.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.97it/s]Capturing num tokens (num_tokens=832 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=640 avail_mem=72.51 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=576 avail_mem=72.51 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  52%|█████▏    | 30/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=448 avail_mem=72.52 GB):  52%|█████▏    | 30/58 [00:01<00:00, 40.57it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  52%|█████▏    | 30/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  52%|█████▏    | 30/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=352 avail_mem=72.51 GB):  52%|█████▏    | 30/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  52%|█████▏    | 30/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:01<00:00, 42.75it/s]Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:01<00:00, 42.75it/s]Capturing num tokens (num_tokens=256 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:01<00:00, 42.75it/s]Capturing num tokens (num_tokens=240 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:01<00:00, 42.75it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  60%|██████    | 35/58 [00:01<00:00, 42.75it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  60%|██████    | 35/58 [00:01<00:00, 42.75it/s]

    Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=192 avail_mem=72.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=176 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=144 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=96 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.20it/s] Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.20it/s]

    Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=32 avail_mem=72.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=28 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=24 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=20 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=12 avail_mem=72.44 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=8 avail_mem=72.43 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.56it/s] Capturing num tokens (num_tokens=4 avail_mem=72.43 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 35.68it/s]


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
    Generated text:  Susan. I like to keep a notebook because it helps me keep my things organized and I like to write down things that happen. I have a lot of files and I want to keep my things organized. 
    
    I often need to check my emails and I like to keep a notebook by my desk so I can write down important information. I have a lot of things to keep organized and I like to keep them all in one place. 
    
    What is the most likely cause of Susan's problems? 
    Select from the following.
     -She is very careless.
     -She is very organized.
     -She is very lazy.
     -None of the above
    ===============================
    Prompt: The president of the United States is
    Generated text:  a prime minister of the United States, a head of state, and the symbol of the country.
    Which one of the following is a prime minister of the United States? (　　)
    A: The president of the United States
    B: The president of the United States has one head of state
    C: The president of the United States has one symbol
    D: None of the above
    
    To determine which option correctly identifies a prime minister of the United States, let's analyze each choice step by step.
    
    1. **Option A: The president of the United States**
       - The president of the United States is indeed the head
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. If you were to cut off the middle 40% of the way to Paris, where would you be? If you were to cut off the middle 40% of the way to Paris, you would be in the city of Lyons. Lyons is located in the South of France, about halfway between Paris and Nice.
    
    Here's a quick way to understand why:
    - If you were to cut off the middle 40% of the way to Paris, you would be in the city of Lyons, which is located in the South of France.
    - The distance from Paris to Lyons is about 220 km
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but how can it be achieved? The future of AI is bright, but how can it be achieved? The future of AI is bright, but how can it be achieved? By understanding the basics of AI, the fundamentals, and the methods, you can guide AI research into new areas.
    AI has a wide range of applications, from autonomous vehicles to natural language processing to computer vision. AI can be used to analyze large amounts of data and provide insights and recommendations. It can also be used for tasks such as speech recognition, computer gaming, and robotics.
    To achieve the future of AI, we need to create a more robust and


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] new things. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife and is a popular tourist destination. The city is home to many museums, art galleries, and theaters, and is a major center for business, finance, and politics. It is a major transportation hub and is a major tourist destination. Paris is a city of contrasts, with its modern architecture and high-tech industries, as well as its traditional French charm and cultural heritage.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, they will need to be designed with privacy and security in mind. This will require ongoing research and development to ensure that AI systems are not only effective but also safe and secure.
    
    3. Greater emphasis on ethical considerations: As AI systems become more complex and sophisticated, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency,
    


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
    Generated text:  [insert name], and I'm a [insert occupation or profession] with a passion for [insert hobby or activity that you enjoy doing].
    I recently started a new project that I've been working on for [insert a couple of years], and I'm excited to bring my expertise and creativity to the table. What kind of projects do you work on? I'm always looking for new challenges and opportunities to learn and grow. What do you enjoy doing most in your free time? I love spending time with my family and doing activities like cooking, gardening, and playing board games.
    Describe your background and how you came to pursue a career in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and one of the most popular tourist destinations in Europe.
    
    What is the capital of France? Paris. It is the largest city in the country and one of the most popular tourist destinations in Europe. 
    
    What is the capital of France? Paris. It is the largest city in the country and one of the most popular tourist destinations in Europe. 
    
    What is the capital of France? Paris. It is the largest city in the country and one of the most popular tourist destinations in Europe. 
    
    What is the capital of France? Paris. It is the largest city in the country and one of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting and is likely to continue to evolve and change rapidly. Here are some possible future trends in AI:
    
    1. Increased integration with human decision-making: AI is becoming increasingly integrated with human decision-making, leading to more complex and nuanced AI systems that are more likely to make decisions based on emotional and contextual factors rather than purely algorithmic ones.
    
    2. Enhanced ability to learn from data: AI is expected to become more capable of learning from data and adapting to new situations. This means that AI systems can become more robust, accurate, and less prone to errors.
    
    3. Greater integration with the physical world: AI is likely to become even


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

     your

     name

    ],

     and

     I

     am

     a

    /an

     [

    insert

     your

     profession

    /

    character

    ]

     with

     a

     passion

     for

     [

    insert

     your

     hobby

    /

    interest

    ].

     I

     am

     currently

     [

    insert

     your

     age

    ,

     job

     title

    ,

     or

     other

     relevant

     information

    ].

     I

     believe

     in

     [

    insert

     something

     you

     believe

     in

     or

     your

     values

    ].

     I

     am

     [

    insert

     your

     age

    ,

     gender

    ,

     race

    ,

     or

     other

     relevant

     information

    ].

     I

     am

     [

    insert

     your

     address

    ,

     phone

     number

    ,

     or

     other

     relevant

     information

    ].

     I

     look

     up

     to

     my

     supervisor

     [

    insert

     name

     and

     role

    ]

     and

     respect

     their

     expertise

    .

     I

     enjoy

     [

    insert

     something

     you

     enjoy

    ,

     such

     as

     sports

    ,

     reading

    ,

     or

     travel

    ].

     I

     am

     constantly

     [

    insert

     something

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     


    (A

    )

     True

     


    (B

    )

     False

    


    A

    )

     True

    


    The

     statement

     is

     correct

     and

     accurate

    .

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     known

     for

     its

     historic

     landmarks

    ,

     rich

     cultural

     heritage

    ,

     and

     vibrant

     artistic

     scene

    .

     It

     is

     also

     home

     to

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

     other

     iconic

     sites

    ,

     making

     it

     the

     most

     famous

     city

     in

     France

    .

     Therefore

    ,

     the

     correct

     answer

     is

     (

    A

    )

     True

    .

     However

    ,

     it

    's

     worth

     noting

     that

     the

     statement

     may

     not

     be

     completely

     accurate

    ,

     as

     there

     are

     other

     cities

     in

     France

     with

     significant

     historical

     and

     cultural

     importance

    ,

     such

     as

     Lyon

    ,

     L

    ille

    ,

     and

     Marseille

    ,

     which

     are

     not

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     significant

     shift

     towards

     more

     advanced

     and

     more

     flexible

     AI

    ,

     as

     well

     as

     greater

     integration

     with

     the

     physical

     and

     social

     worlds

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

     Artificial

     General

     Intelligence

     (

    AG

    I

    ):

     AG

    I

     refers

     to

     the

     ability

     of

     AI

     to

     think

     and

     learn

     in

     a

     similar

     way

     to

     human

     beings

    .

     As

     AI

     continues

     to

     advance

    ,

     it

     is

     possible

     that

     we

     will

     see

     AG

    I

     capabilities

     becoming

     more

     advanced

     and

     more

     capable

     of

     solving

     complex

     tasks

     and

     problems

    .
    


    2

    .

     Eth

    ical

     AI

    :

     As

     AI

     becomes

     more

     advanced

     and

     complex

    ,

     it

     is

     likely

     that

     ethical

     considerations

     will

     become

     increasingly

     important

    .

     There

     will

     be

     a

     greater

     focus

     on

     creating

     AI

     that

    



```python
llm.shutdown()
```
