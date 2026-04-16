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
    [2026-04-16 22:04:16] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.88it/s]


    2026-04-16 22:04:20,834 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 22:04:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:02<00:06,  6.77it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.77it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.77it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.77it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.81it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 30.04it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 37.76it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 37.76it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 47.26it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 47.26it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 47.26it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 47.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:05, 10.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:05, 10.47it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:05, 10.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.73 GB):   7%|▋         | 4/58 [00:00<00:04, 13.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   7%|▋         | 4/58 [00:00<00:04, 13.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.72 GB):   7%|▋         | 4/58 [00:00<00:04, 13.10it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.72 GB):   7%|▋         | 4/58 [00:00<00:04, 13.10it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.72 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.20it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.39it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.57it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.57it/s]

    Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.48it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.48it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.48it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.48it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.48it/s]Capturing num tokens (num_tokens=448 avail_mem=120.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.48it/s]Capturing num tokens (num_tokens=448 avail_mem=120.24 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.50it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.50it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.50it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.50it/s]

    Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.50it/s]Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.50it/s]Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=208 avail_mem=120.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=192 avail_mem=120.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=192 avail_mem=120.24 GB):  71%|███████   | 41/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=176 avail_mem=119.70 GB):  71%|███████   | 41/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  71%|███████   | 41/58 [00:01<00:00, 41.91it/s]

    Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  71%|███████   | 41/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  71%|███████   | 41/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  71%|███████   | 41/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.09it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.30it/s]

    Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.18it/s] Capturing num tokens (num_tokens=4 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:01<00:00, 35.75it/s]


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
    Generated text:  Andrew and I'm a world-class software engineer. I'm a little rusty in coding, so I don't code all that much. However, I do know a lot of programming languages and I've been helping others learn JavaScript. I was wondering if you could assist with a few JavaScript tasks. Here are some tasks that I would like you to assist with: 
    
    1. Write a program that takes a string of digits and returns the largest digit in the string. If there are multiple largest digits, return the first one.
    
    For example, if the input is "234123", the output should be "4".
    
    2
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military jets to buy. He can buy 100 jets now, or 50 jets next year, or 100 jets in 5 years, or 300 jets in 10 years. Each jet costs $500,000. If the president's goal is to have 500 jets by the end of 10 years, how much would he need to spend on these jets?
    To determine the total cost of purchasing 500 jets by the end of 10 years, we need to calculate the cost for each scenario and then
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. ( )
    A: Correct
    B: Incorrect
    C: 
    D: 
    To determine whether the statement "The capital of France is Paris" is correct or incorrect, we need to recall the location of Paris as the capital of France.
    
    1. **Identify the capital of France:**
       The capital of France is Paris. This is a well-known fact in the world of European history.
    
    2. **Verify the statement:**
       The statement claims that Paris is the capital of France. Since Paris is indeed the capital of France, this statement is correct.
    
    3. **Conclusion:**
       Based on the information provided
    ===============================
    Prompt: The future of AI is
    Generated text:  here, but it's not here yet. But in the coming decade, it is going to be for sure. Technology companies are going to revolutionize the way we live our lives, and it is coming at us from all angles. It's not going to be easy, but it's entirely possible.
    AI is becoming more and more ubiquitous in our daily lives. The idea of a robot and computer program that can do everything from making decisions to writing poetry, playing chess, and playing musical instruments, is here now.
    The technology will enable us to perform tasks that are currently performed by humans, such as diagnosing diseases, creating new drugs


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Character] who has always been [Positive Traits]. I am [Positive Traits] and I am [Positive Traits]. I am [Positive Traits] and I am [Positive Traits]. I am [Positive Traits] and I am [Positive Traits]. I am [Positive Traits] and I am [Positive Traits]. I am [Positive Traits] and I am [Positive Traits]. I am [Positive Traits] and I am [Positive Traits]. I am [Positive Traits] and I am [Positive Traits]. I am [Positive Traits
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is accurate and provides a brief overview of the capital city's location and significance in French culture and politics. It is a widely recognized and well-known city in the world, known for its rich history, beautiful architecture, and vibrant culture. Paris is the capital of France and is home to the country's political, economic, and cultural center. It is also a major tourist destination, attracting millions of visitors each year. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, as well as its diverse and multicultural population. Paris is a city that
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI becomes more advanced, it is likely to be used in even more
    


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
    Generated text:  [Character's Name] and I'm a [fill in the blank] year-old software engineer from [city or location]. I'm passionate about [mention something about your field of expertise or hobbies]. I'm always up for learning new things and have a fun-loving personality. I enjoy exploring different cultures and trying out new recipes. What is your favorite hobby or activity to do at home?
    Hello, my name is [Character's Name] and I'm a [fill in the blank] year-old software engineer from [city or location]. I'm passionate about [mention something about your field of expertise or hobbies]. I'm always up for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic architecture, vibrant culture, and important historical landmarks. It serves as the political, cultural, and economic center of the country and is home to numerous renowned museums, art galleries, and historical landmarks, including the Louvre and the Eiffel Tower. Despite its international status, Paris remains a cultural and political hub within the country. It's a must-visit for anyone seeking an immersive experience of French culture and history. Paris is also known as the "City of Light" and a global cultural hub, drawing visitors from around the world. It is one of the most visited cities in the world, with many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a variety of trends, including:
    
    1. Increased AI integration with existing technologies: AI will continue to be integrated into other technologies such as voice assistants, self-driving cars, and wearable devices, making them more intuitive and convenient to use.
    
    2. More autonomous and self-driving vehicles: AI will continue to improve, and autonomous vehicles will become more common, allowing for more convenient and efficient transportation.
    
    3. AI-powered healthcare advancements: AI will continue to advance in the medical field, with applications in diagnostics, drug development, and personalized medicine.
    
    4. AI for education: AI will continue to be used in education, with


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

     name

    ].

     I

     am

     an

     individual

     with

     a

     deep

     love

     for

     storytelling

    ,

     and

     I

     have

     a

     knack

     for

     weaving

     stories

     that

     are

     both

     captivating

     and

     rel

    atable

    .

     I

     am

     passionate

     about

     using

     my

     skills

     to

     bring

     characters

     to

     life

     and

     help

     people

     discover

     their

     own

     paths

     in

     the

     world

    .

     I

     believe

     that

     storytelling

     is

     a

     powerful

     tool

     for

     growth

     and

     exploration

    ,

     and

     I

     am

     eager

     to

     explore

     new

     ways

     of

     sharing

     my

     passion

     for

     writing

    .

     Thank

     you

     for

     having

     me

    .

     Let

    's

     connect

     and

     explore

     more

     together

    .

     [

    Insert

     name

    ]

     [

    Your

     name

    ]

     [

    Company

    ]

     [

    Insert

     job

     title

    ]

     [

    Insert

     position

    ]

     [

    Insert

     company

    ]

     [

    Insert

     company

    ]

     [

    Insert

     company

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

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

     It

     is

     also

     a

     major

     financial

     and

     cultural

     hub

    ,

     hosting

     numerous

     world

    -class

     museums

    ,

     theaters

    ,

     and

     museums

    .

     Additionally

    ,

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     known

     for

     its

     beautiful

     architecture

     and

     stunning

     views

     of

     the

     city

    .

     It

     is

     the

     second

    -largest

     city

     in

     France

    ,

     with

     a

     population

     of

     around

     

    1

    1

     million

     people

    .

     Paris

     is

     a

     bustling

     city

     with

     a

     diverse

     and

     vibrant

     culture

    ,

     making

     it

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     France

    's

     history

    ,

     art

    ,

     cuisine

    ,

     and

     more

    .

     It

     is

     also

     known

     as

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     hardware

     and

     software

    ,

     new

     developments

     in

     machine

     learning

     and

     natural

     language

     processing

    ,

     and

     shifts

     in

     societal

     values

     and

     norms

    .

     Some

     possible

     trends

     include

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     into

     a

     variety

     of

     industries

    ,

     it

     is

     likely

     to

     interact

     more

     closely

     with

     other

     technologies

    ,

     such

     as

     the

     Internet

     of

     Things

     (

    Io

    T

    )

     and

     the

     cloud

    .
    


    2

    .

     Increased

     use

     of

     AI

     for

     autonomous

     decision

    -making

    :

     Autonomous

     AI

     systems

     will

     become

     more

     common

    ,

     allowing

     for

     more

     autonomous

     decision

    -making

     in

     industries

     such

     as

     transportation

    ,

     healthcare

    ,

     and

     manufacturing

    .
    


    3

    .

     Greater

     emphasis

     on

     privacy

     and

     security

    



```python
llm.shutdown()
```
