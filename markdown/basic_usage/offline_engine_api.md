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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.52it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.50it/s]


    2026-04-09 12:49:43,131 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 12:49:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.70it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.07it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.18it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.18it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.18it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.18it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.18it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.18it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.18it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.18it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.78it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.78it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.78it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.78it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.78it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.78it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.78it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.10it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.10it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.10it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.10it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.10it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.10it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.10it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.35it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.35it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.35it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.35it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.35it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.35it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.35it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.35it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 47.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.74it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.90it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 29.83it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 29.83it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 29.83it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 29.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.13it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.13it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.13it/s] Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.68it/s]Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.68it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.68it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.68it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.68it/s]

    Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  45%|████▍     | 26/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:00<00:00, 31.85it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:00<00:00, 31.85it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.85it/s]

    Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  60%|██████    | 35/58 [00:01<00:00, 35.46it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 35.46it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 35.46it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 35.46it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 35.46it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  60%|██████    | 35/58 [00:01<00:00, 35.46it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.41it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.41it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.41it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.41it/s]Capturing num tokens (num_tokens=144 avail_mem=120.23 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.41it/s]Capturing num tokens (num_tokens=128 avail_mem=120.23 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.41it/s]

    Capturing num tokens (num_tokens=128 avail_mem=120.23 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.40it/s]Capturing num tokens (num_tokens=112 avail_mem=119.84 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.40it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.40it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.40it/s]Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.40it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.40it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.32it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.32it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.32it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.32it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.32it/s]

    Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.32it/s]Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.31it/s] Capturing num tokens (num_tokens=4 avail_mem=118.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:01<00:00, 36.14it/s]


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
    Generated text:  Shu and I am a 14 year old female. I have no symptoms, I am healthy. I have been able to stay in the house for many months without going out and I have not done anything that I think might cause someone to think I am sick. I do not have any allergies or other chronic illnesses. I have been mostly vegetarian, that is I have never eaten meat. My favorite food is broccoli. I have never been sick and have never been hospitalized. I have a pet rabbit. I was wondering if there are any foods that are supposed to be okay for people with symptoms of the flu or not to worry
    ===============================
    Prompt: The president of the United States is
    Generated text:  in the White House, but he is not a politician and does not hold a position at the federal government. Which of the following cannot be his job title?
    
    A) Chief Justice of the United States
    
    B) President of the United States
    
    C) Secretary of State
    
    D) President of the United States
    To determine which of the given options cannot be the president's job title, let's analyze each option step by step.
    
    1. **Chief Justice of the United States**: This is the highest judicial position in the United States, and it is typically held by the president. Therefore, it cannot be the president's job title.
    
    
    ===============================
    Prompt: The capital of France is
    Generated text: : A: Paris B: Lyon C: Nice D: Toulouse\nThe capital of France is: A: Paris\n\nAn information-based answer is provided below the list of options. An explanation for choosing A as the answer is given after the list of options.
    ===============================
    Prompt: The future of AI is
    Generated text:  on the horizon, and it's not just about developing more advanced algorithms or gadgets. As AI becomes more integrated into our daily lives, it's also transforming how we interact with our environment and the people around us.
    One area where AI is making a significant impact is in the healthcare industry. In the past, AI was mainly used to analyze large amounts of data and extract insights, but now it's being used to develop more accurate and personalized medical treatments. For example, AI algorithms can analyze medical images to detect early signs of disease, predict patient outcomes, and even assist doctors in making decisions about treatment options.
    But AI is not just a


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I love [book/movie]. I'm always looking for new ways
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and music, and is a major tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. The city is home to many famous museums, art galleries, and theaters, and is a hub for culture and entertainment. Paris is a city of art, science,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from their environment and improve their performance over time.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and responsible AI development. This could involve developing AI systems that
    


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
    Generated text:  [Name]. I am a skilled [insert skill or expertise]. I love [insert hobby or passion]. I'm a [insert personality trait] and I am passionate about [insert something]. I enjoy [insert something]. What brings you to this world?
    Hello, my name is [Name]. I am a skilled [insert skill or expertise]. I love [insert hobby or passion]. I'm a [insert personality trait] and I am passionate about [insert something]. I enjoy [insert something]. What brings you to this world? (This character is a scientist, and they are excited about the idea of exploring the universe and uncovering
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Ville", the "City of a Hundred Rivers". 
    
    The city is located in the center of the Mediterranean Sea, on the banks of the River Seine. It is home to the Louvre Museum, the most famous of its kind in the world, and the Notre Dame Cathedral. Paris is also known for its jazz music, its food, and its art. It is one of the most important cultural and historical centers in the world. It is often referred to as the "City of Light" and "The City of Light". 
    
    The city of Paris is a UNESCO World Heritage site and a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a number of trends and developments that will shape the way we interact with technology, work, and even our families and communities. Here are some possible future trends in AI:
    
    1. Increased automation: AI will continue to automate a growing number of tasks that are currently performed by humans, such as data analysis, natural language processing, and decision-making. This will lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI will become more interconnected: AI will continue to be integrated into everyday life, from smart homes to autonomous vehicles, and will become increasingly embedded in our daily


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

    ...

     [

    insert

     the

     name

     of

     the

     character

     here

    ].

     I

    'm

     a

     [

    insert

     occupation

     or

     profession

    ].

     I

     enjoy

     [

    insert

     something

     that

     brings

     me

     joy

     or

     makes

     me

     happy

    ].

     My

     favorite

     hobby

     is

     [

    insert

     something

     that

     reminds

     me

     of

     my

     personality

     or

     interests

    ].

     What

    's

     your

     name

    ,

     [

    insert

     your

     name

     here

    ]?

     I

    'm

     so

     excited

     to

     meet

     you

    !

     Let

    's

     chat

    !

     [

    insert

     how

     you

     met

     the

     person

    ,

     any

     trivia

    ,

     etc

    .

    ].

     I

     hope

     you

     have

     a

     great

     day

    !

     [

    insert

     a

     friendly

     greeting

     and

     inquire

     about

     the

     day

     or

     time

    ].

     [

    insert

     how

     you

     met

     the

     person

    ,

     any

     trivia

    ,

     etc

    .

    ].

     Hi

     [

    insert

     your

     name

     here

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     of

     light

     and

     learning

    ,

     where

     the

     E

    iff

    el

     Tower

     stands

     as

     a

     symbol

     of

     France

     and

     one

     of

     the

     world

    's

     most

     famous

     landmarks

    .

     It

    's

     a

     bustling

     met

    ropolis

     with

     a

     rich

     cultural

     heritage

    ,

     making

     it

     a

     popular

     tourist

     destination

     and

     a

     cultural

     center

    .

     
    


    Paris

     is

     also

     known

     for

     its

     gastr

    onomic

     delights

    ,

     with

     its

     famous

     French

     cuisine

    ,

     including

     cro

    iss

    ants

    ,

     b

    oud

    in

    ,

     and

     esc

    arg

    ot

    ,

     as

     well

     as

     its

     romantic

     architecture

     and

     fashion

     scene

    .

     
    


    Paris

     is

     home

     to

     several

     world

    -f

    amous

     museums

    ,

     including

     the

     Lou

    vre

    ,

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     Centre

     Pom

    pid

    ou

    ,

     as

     well

     as

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     with

     numerous

     trends

     shaping

     the

     technology

    's

     direction

     and

     impact

     on

     society

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Transparency

     and

     Accountability

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     there

     will

     be

     greater

     demand

     for

     transparency

     and

     accountability

     in

     their

     design

     and

     operation

    .

     This

     will

     require

     developers

     and

     researchers

     to

     become

     more

     transparent

     about

     their

     algorithms

     and

     decision

    -making

     processes

    ,

     and

     to

     work

     towards

     improving

     transparency

     in

     other

     areas

     of

     AI

     as

     well

    .
    


    2

    .

     AI

     Integration

     with

     Human

     Intelligence

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

     there

     will

     be

     a

     greater

     emphasis

     on

     the

     ability

     of

     AI

     to

     learn

     and

     adapt

     to

     new

     information

     and

     context

    .

     This

     will

     require

     developers

    



```python
llm.shutdown()
```
