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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.12it/s]


    2026-04-28 07:54:05,955 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 07:54:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:28,  4.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:28,  4.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:28,  4.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:28,  4.72s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:28,  4.72s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.13it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.13it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.17it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 39.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 17.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 17.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 17.32it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 17.32it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.40 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.39 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.09it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.09it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.09it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.82it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 28.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 28.82it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 28.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  21%|██        | 12/58 [00:00<00:01, 28.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.01it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.32it/s]Capturing num tokens (num_tokens=960 avail_mem=137.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.32it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.32it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.32it/s]Capturing num tokens (num_tokens=768 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.32it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.32it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.22it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.22it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.22it/s]

    Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.22it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.22it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.92it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.92it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.92it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.92it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.92it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.99it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.99it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.99it/s]

    Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.99it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.99it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=176 avail_mem=137.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.85it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.85it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.85it/s]

    Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.85it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.85it/s] Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  81%|████████  | 47/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  81%|████████  | 47/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  81%|████████  | 47/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  81%|████████  | 47/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  81%|████████  | 47/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  81%|████████  | 47/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.09it/s]

    Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.09it/s] Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.93it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.93it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 35.83it/s]


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
    Generated text:  Z and I am a 15 year old male with no severe medical condition. I have been diagnosed with a 45 degree bend in my left leg, and I have been told that I need to be hospitalized and have a cast put on. What are the risks and complications I could be experiencing if I do not have a cast on? What are the benefits of having a cast on for me?
    The risk of complications associated with a cast is very low. A cast will help to immobilize the leg and help to reduce swelling. This can help to prevent further damage to the blood vessels and nerves in the leg.
    The benefit
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by a 2-person committee. The president's running mate is represented by a 2-person committee. In how many ways can all the committees be combined into a presidential and vice-presidential choice?
    
    To determine the total number of ways to combine the two committees into a presidential and vice-presidential choice, we need to consider the following:
    
    1. There are 2 people who can be the president of the United States.
    2. There are 2 people who can be the running mate of the president.
    
    First, we calculate the number of ways to choose the president. Since there are 2 people who can be the president,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. According to the 2019 data, the population of Paris is 2.1 million. The area of Paris is 109.7 square kilometers. Calculate the average population density of Paris.
    To calculate the average population density of Paris, we need to follow these steps:
    
    1. **Identify the total population of Paris:**
       The total population of Paris is given as \(2.1\) million.
    
    2. **Identify the area of Paris:**
       The area of Paris is given as \(109.7\) square kilometers.
    
    3. **Calculate the population density:**
      
    ===============================
    Prompt: The future of AI is
    Generated text:  in the making, as more and more data is produced and available to be mined for use by businesses and governments. But we do not yet know how much of that data is useful, and how much is surplus.
    The big data challenge is that even if we have a large amount of data, it can be difficult to find and extract the useful information. Another big problem is that while we can extract data, we don't know how to use it to improve the performance of our businesses and governments.
    The research team at the KDD workshop and NIPS conference is trying to tackle these problems. They are developing new approaches to extract useful information


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? As a [Job Title], I'm always looking for ways to improve my skills and knowledge. I'm always eager to learn new things and try new things. I'm also a great communicator and enjoy working with people. What's your favorite hobby or activity? As a [Job Title], I enjoy spending time with my family and friends, reading, and playing sports. What's your favorite book or movie? As a [Job Title], I love reading and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in Europe by population. It is located on the Seine River and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is known for its rich history, art, and culture, and is a popular tourist destination for visitors from around the world. The city is also home to many important institutions and organizations, including the French Academy of Sciences and the French National Library. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to inspire and captivate people from all over the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology continues to improve, we can expect to see even more widespread use of AI in finance.
    
    3. Increased use of AI in transportation: AI
    


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
    Generated text:  [name], and I'm here to help you with any questions you have about [industry]. I'm confident in my abilities and I'm always here to assist you. I'm [age], and I've been in the [industry] industry for [number of years] years now. I enjoy [specific interest or expertise] and I'm always learning and growing. So, if you have any questions about [industry], feel free to ask and I'll do my best to answer them. Thank you for your time. 
    [Your Name] is a self-proclaimed [industry] enthusiast with over [number of years] years of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France. It is located in the south of the country, in the Île-de-France region. The city is known for its rich history, stunning architecture, and vibrant culture. Paris is a major center for art, fashion, and science, and is the largest city in the European Union by population. It is also known as the "City of Love" due to its love-ins and romance-festivals. The city is home to several famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a must-visit destination for anyone
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising and exciting, with several potential trends shaping its development. Some of the key trends include:
    
    1. Increased AI Transparency: As AI systems become more advanced, we may see an increase in transparency and explainability. This means that AI systems will be more accessible and understandable to users, allowing them to make informed decisions based on the data and patterns presented to them.
    
    2. Enhanced Personalization: With the help of AI, we can create personalized experiences for users. AI can analyze user data and preferences to provide tailored recommendations and suggestions.
    
    3. Increased AI Ethics and Accountability: As AI systems become more sophisticated, they will need to be held


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

     character

    's

     name

    ],

     and

     I

    'm

     a

     [

    insert

     occupation

     or

     profession

    ]

     from

     [

    insert

     location

    ].

     I

     enjoy

     [

    insert

     one

     or

     two

     personal

     traits

     or

     hobbies

     that

     make

     me

     who

     I

     am

    ].

     I

     believe

     in

     [

    insert

     what

     you

     believe

     makes

     you

     a

     great

     leader

     or

     manager

    ],

     and

     I

     strive

     to

     [

    insert

     a

     specific

     goal

     or

     goal

     that

     I

     believe

     in

    ].

     
    


    And

     in

     summary

    ,

     I

    'm

     [

    insert

     one

     or

     two

     personality

     traits

     or

     qualities

     that

     make

     me

     stand

     out

    ].

     Thank

     you

     for

     asking

     to

     meet

     me

    ,

     and

     I

     look

     forward

     to

     our

     conversation

    !

     #

    Meet

    The

    Person

     #

    Character

    Int

    roduction

    


    #

    Fre

    el

    ance

    Writer

     #

    Author

     #

    Ad

    vent

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    .

     True

    


    B

    .

     False

    
    


    B

    .

     False

    


    You

     are

     an

     AI

     assistant

     that

     helps

     people

     find

     information

    .

     Don

    't

     use

     me

     to

     create

    ,

     copy

     or

     create

     a

     story

    .

     My

     own

     true

     stories

     is

     Paris

    .

     I

     can

    't

     tell

     you

     about

     your

     own

     real

     life

    .

     If

     you

     use

     it

     for

     something

     else

    ,

     please

     let

     me

     know

    .

     I

    'll

     be

     happy

     to

     help

    .

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     access

     to

     real

    -time

     information

     about

     specific

     locations

     or

     events

    .

     However

    ,

     I

     can

     help

     you

     with

     factual

     information

     about

     Paris

    .

     Let

     me

     know

     if

     you

     need

     any

     other

     help

    !

     #

    Paris

     #

    F

    actual

     #

    Information

     #

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     exciting

    ,

     with

     endless

     possibilities

     for

     technological

     advancement

     and

     transformation

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     that

     are

     likely

     to

     shape

     the

     industry

     in

     the

     coming

     years

    :
    


    1

    .

     Autonomous

     vehicles

    :

     As

     the

     need

     for

     safer

     and

     more

     efficient

     transportation

     continues

     to

     grow

    ,

     autonomous

     vehicles

     are

     likely

     to

     become

     increasingly

     popular

    .

     This

     technology

     would

     use

     AI

     to

     improve

     road

     safety

    ,

     reduce

     traffic

     congestion

    ,

     and

     increase

     efficiency

     in

     logistics

     and

     delivery

    .
    


    2

    .

     Artificial

     general

     intelligence

    :

     With

     the

     advances

     in

     AI

    ,

     it

     is

     possible

     that

     we

     may

     one

     day

     be

     able

     to

     harness

     the

     power

     of

     AI

     to

     understand

     and

     interpret

     human

    -like

     language

    .

     This

     would

     enable

     machines

     to

     think

     and

     learn

     in

     a

     way

     that

    



```python
llm.shutdown()
```
