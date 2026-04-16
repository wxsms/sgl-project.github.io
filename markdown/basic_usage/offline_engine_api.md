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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-16 08:52:41] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.72it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.71it/s]


    2026-04-16 08:52:47,339 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 08:52:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.68it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.68it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.68it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  6.68it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.68it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.68it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.68it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.68it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.68it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.68it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.68it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.68it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.68it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.68it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.68it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.68it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.68it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.68it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.68it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.68it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.57it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.57it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.57it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.57it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.57it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.57it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.57it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.57it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.57it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.57it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.48it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.48it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.48it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.48it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.48it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.48it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.48it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.48it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.48it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.48it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 50.06it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 17.44it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.44it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.44it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 19.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 19.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 19.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 19.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.04it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.04it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.99it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.99it/s] Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.98it/s]Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.98it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.98it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.98it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.98it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.98it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.37it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.37it/s]

    Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.37it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.37it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.37it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.37it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.41it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.41it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.41it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.41it/s]

    Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.89it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.89it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.89it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.89it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.89it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.89it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  71%|███████   | 41/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  71%|███████   | 41/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 39.33it/s]

    Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 39.33it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.87it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.51it/s]

    Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.12it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 35.84it/s]


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
    Generated text:  22 years old, 5' 11" tall, and my weight is 175 lbs. I'm married and have a 5 yr old child. I also have a 16 yr old son that I am single. I have a baby in my 22 year old daughter and am currently 25 weeks pregnant. My ex partner filed for divorce in January of 2018. I am trying to get back in to dating again and I have been taking medication for depression. I have been on medication since last year. I want to get my life back on track. I want
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected by a single electoral college consisting of the states and territories with voting age requirements greater than 30. In the last election, 5 states voted in favor of a bill that passed the House and Senate. There were 22 states that voted against the bill, with 12 of those states requiring a higher voting age than 30. How many states voted in favor of the bill?
    
    Let's break down the problem step by step:
    
    1. **Determine the total number of states voting in favor of the bill**: This is given as 5 states.
    2. **Determine the total number of states voting
    ===============================
    Prompt: The capital of France is
    Generated text:  located in which of the following? ____
    A. Provence-Alpes-Côte d'Azur
    B. English Channel
    C. Alps
    D. The Rhine
    Answer:
    A
    
    A corporate bank is a type of ( )
    A. Financial institution
    B. Non-financial institution
    C. Corporate entity
    D. Financial intermediary
    Answer:
    A
    
    Which of the following is NOT a true confession? 
    A. A confession made under torture
    B. A confession made to a spy
    C. A confession made to a court in another country
    D. A confession made to a bank
    Answer:
    D
    ===============================
    Prompt: The future of AI is
    Generated text:  decentralized, and it is here to stay.
    
    With the introduction of blockchains, the future of AI is decentralized.
    
    In a decentralized world, data is distributed among several entities. This is the core concept of a blockchain.
    
    The term blockchain is an acronym for the words "block" and "chain," which are connected. This allows for a transaction to be verified with all transactions in a chain, ensuring transparency and security. This decentralized data is what we are referring to as "AI." Every transaction in the chain is verified by everyone in the network.
    
    Because of this, it is possible to track the value of every transaction, and every block


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast and I love to [Describe an activity or hobby you enjoy]. I'm always looking for new experiences and learning new things, and I'm always eager to share my knowledge with others. I'm a [Favorite Color] person and I love to [Describe a favorite activity or hobby]. I'm always looking for new challenges and opportunities to grow and learn, and I'm always eager to share my experiences with others. I'm a [Favorite Book] lover and I love to [Describe a favorite book or movie].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the country and the second-largest city in the European Union. It is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also known for its rich cultural heritage, including its museums, theaters, and art galleries, and its vibrant nightlife. The city is a major transportation hub, with many major highways and rail lines passing through its heart. Paris is a popular tourist destination, with millions of visitors each year, and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends that could be expected in the coming years:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be an increased focus on ethical considerations. This could include issues such as bias, transparency, and accountability. AI developers will need to be more mindful of how their technology is used and ensure that it is used in a way that is fair and just.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence in the future.
    


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
    Generated text:  [name] and I'm [age]. I'm a [career field] with [number of years of experience]. I have a passion for [something specific] that drives me to [something]. I'm excited to meet you and discuss what kind of projects or projects you're working on. Have you done any projects before? How can I help you? What's your project? What's your story about? Please provide me with a brief summary of your project. Good luck. Let me know when we can connect. Hello, my name is [name] and I'm [age]. I'm a [career field] with [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, and is considered one of the oldest cities in the world. It is located on the Loire River and is the seat of government, the most populous city in Europe, and the largest city in metropolitan France. The city is home to numerous museums, theaters, and cultural institutions, and is a major economic and cultural center. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, as well as its delicious cuisine, vibrant nightlife, and arts scene. The city is a hub for education, research, and innovation, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to evolve and advance, driven by a combination of technological progress, economic growth, and societal changes. Here are some possible trends that could shape AI in the coming years:
    
    1. Increased focus on ethical considerations: As AI becomes more ubiquitous and integrated into our daily lives, there will be a greater emphasis on ethical considerations. This could include issues like bias, transparency, accountability, and privacy.
    
    2. Greater adoption of AI in areas like healthcare, finance, and manufacturing: AI is already being used in many industries, and it is likely to be adopted further in the coming years. For example, AI is being used in healthcare


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

    occupation

     or

     role

    ]

     who

     loves

     [

    occupation

     or

     role

    ]

     in

     [

    job

     title

    ].

     I

     enjoy

     spending

     time

     [

    reason

    ]

     and

     I

     enjoy

     [

    reason

    ].

     In

     my

     free

     time

    ,

     I

     enjoy

     [

    reason

    ]

     and

     [

    reason

    ].

     I

     am

     a

     [

    character

     trait

     or

     personality

    ]

     who

     [

    reason

    ].

     I

     believe

     that

     [

    reason

    ]

     is

     important

     to

     me

    ,

     and

     I

     try

     to

     [

    reason

    ]

     to

     achieve

     it

    .

     What

     is

     your

     name

    ,

     and

     what

     do

     you

     do

    ?

     I

    'm

     [

    Name

    ]

     and

     I

     love

     [

    occupation

     or

     role

    ].

     I

     enjoy

     [

    reason

    ]

     and

     I

     enjoy

     [

    reason

    ].

     In

     my

     free

     time

    ,

     I

     enjoy

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     This

     city

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

     and

     the

     Lou

    vre

     Museum.

     It

     is

     also

     a

     center

     for

     French

     culture

    ,

     cuisine

    ,

     and

     fashion

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     cultural

     center

    ,

     with

     many

     museums

    ,

     theaters

    ,

     and

     parks

    .

     The

     French

     language

     is

     widely

     spoken

     in

     Paris

    ,

     and

     the

     city

     has

     a

     rich

     history

     and

     culture

     dating

     back

     to

     the

     Roman

     Empire

    .

     The

     capital

     is

     a

     bustling

     met

    ropolis

     with

     a

     diverse

     population

     that

     includes

     many

     ethnic

     groups

     and

     communities

    .

     The

     city

     is

     also

     known

     for

     its

     wine

     industry

    ,

     with

     several

     vine

    yards

     in

     the

     region

    .

     Overall

    ,

     Paris

     is

     a

     fascinating

     and

     culturally

     rich

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     shaped

     by

     a

     variety

     of

     trends

     and

     technologies

    ,

     including

    :
    


    1

    .

     Deep

     learning

    :

     This

     is

     a

     new

     type

     of

     AI

     that

     allows

     computers

     to

     learn

     from

     and

     recognize

     patterns

     in

     large

     amounts

     of

     data

    .

     It

     is

     expected

     to

     be

     used

     in

     areas

     such

     as

     image

     recognition

    ,

     speech

     recognition

    ,

     and

     natural

     language

     processing

    .
    


    2

    .

     Automation

    :

     AI

     is

     already

     being

     used

     in

     a

     variety

     of

     areas

    ,

     from

     manufacturing

     to

     customer

     service

    ,

     and

     it

     is

     expected

     to

     continue

     to

     increase

     in

     complexity

     and

     reach

     wider

     applications

     in

     the

     future

    .
    


    3

    .

     Cloud

     computing

    :

     AI

     will

     continue

     to

     rely

     on

     cloud

     computing

     for

     processing

     power

     and

     storage

    ,

     which

     will

     become

     even

     more

     prevalent

     as

     more

    



```python
llm.shutdown()
```
