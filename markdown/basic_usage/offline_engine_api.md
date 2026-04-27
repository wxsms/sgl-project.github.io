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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.57it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.56it/s]


    2026-04-27 18:15:20,927 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 18:15:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:41,  4.94s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:41,  4.94s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.35it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.35it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.35it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.35it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.35it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.35it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.35it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.35it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:05,  7.35it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 12.34it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 12.34it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 12.34it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 12.34it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 12.34it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 12.34it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 12.34it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:02, 12.34it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:05<00:02, 12.34it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:05<00:02, 12.34it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 19.05it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 19.05it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 19.05it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 19.05it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 19.05it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 19.05it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 19.05it/s]

    Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 19.05it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 19.05it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 19.05it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 26.93it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 36.47it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 36.47it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 36.47it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 36.47it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:03, 17.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:03, 17.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   7%|▋         | 4/58 [00:00<00:03, 17.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   7%|▋         | 4/58 [00:00<00:03, 17.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.20it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.64it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.11it/s] Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  43%|████▎     | 25/58 [00:00<00:00, 37.16it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  43%|████▎     | 25/58 [00:00<00:00, 37.16it/s]

    Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  43%|████▎     | 25/58 [00:00<00:00, 37.16it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  43%|████▎     | 25/58 [00:00<00:00, 37.16it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  43%|████▎     | 25/58 [00:00<00:00, 37.16it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 37.90it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:01<00:00, 37.90it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:01<00:00, 37.90it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.00it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.00it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.00it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.00it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.00it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.03it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.03it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.03it/s]

    Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.03it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.03it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.03it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.98it/s] Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 37.54it/s]

    Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  81%|████████  | 47/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.45it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.45it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.45it/s]Capturing num tokens (num_tokens=16 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.45it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.45it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.45it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.04it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.04it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 33.10it/s]


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
    Generated text:  Sunita. I'm a U.S. citizen. I was born in Burma, a country that's called Myanmar. I've lived in many different countries including India, Afghanistan, Canada, the U.S. and China. I have a master's degree in medicine, and I have had my own company for three years. I'm trying to apply for citizenship in the U.S. but have been told by the government they will not accept me as a U.S. citizen. 
    
    In my application, I have said that I've been a citizen of the United Kingdom since I was a child and had been living there my entire life
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military advisors to have. He knows that there are 52 advisors to allocate to the military. The president wants the ratio of advisors to be as close as possible to 1:20. How many advisors should he have to achieve this ratio?
    To determine the number of advisors the president should have to achieve a ratio of 1:20 to 52, we need to set up the problem using the ratio and then find the number of advisors that will maintain this ratio.
    
    Let's denote the number of advisors by \( x \). According to the problem, the ratio of advisors to be 
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the second largest city in the world, located on the Île de France (Paris l'île) and surrounded by the Seine River. Paris is the capital city of the Île-de-France region, which includes the Île-de-France province, the Hauts-de-France region, and the Picardy region. The city is located on the banks of the Seine River and is the seat of the First Republic, the Second Republic, the Third Republic, and the Fifth Republic.
    Paris is the capital city of France. Paris is the second largest city in the world, located on the Î
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, and the question is what it will look like for humans. By April 2022, 15 years from now, the world will see the first full report of AI in action, and the vast majority of the AI in use will be in development. Will this new technology be well received by society? How will AI affect the jobs of people, and what will the changes be? With AI being so prevalent in many areas, many will be surprised at how it will affect their lives. AI technology is rapidly growing, and its future is uncertain. However, it is expected that it will be in a position to


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] in the field of [Field of Interest]. I'm passionate about [Why I'm Passionate About My Field]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Favorite Hobby] that I enjoy [Number of Hours] per week. I'm a [Favorite Book] that I read every day. I'm a [Favorite Movie] that I watch every weekend. I'm a [Favorite Sport] that I play [Number of Hours] per week
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and culture. It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 5 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, as well as its vibrant arts scene and culinary delights. The city is also home to many world-renowned museums, including the Louvre and the Musée d'Orsay, and is a popular tourist destination for its beautiful architecture and cultural attractions. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be increased pressure to address ethical concerns, such as bias, transparency, and accountability. This could lead to more rigorous ethical standards and regulations
    


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
    Generated text:  [Name]. I'm a [type of job or occupation] who has a passion for [what you love doing most]. What are your hobbies and interests outside of work?
    
    Hello, my name is [Name]. I'm a [type of job or occupation] who has a passion for [what you love doing most]. I'm a [how you are described]. What are your hobbies and interests outside of work?
    
    You might also like... 🌟 I'm a [type of job or occupation] with a passion for [what you love doing most]. I'm a [how you are described]. 🌟 What are your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the second largest city in Europe by population. It is a cultural and historical center of France, known for its rich history, beautiful architecture, and vibrant nightlife. Paris is also a hub for business, finance, and trade, with many international companies headquartered there. The city is home to the Eiffel Tower and many other iconic landmarks, including the Louvre Museum and Notre Dame Cathedral. Paris is known for its romantic, elegant atmosphere and is a popular tourist destination for visitors from around the world. 
    
    Some additional facts about Paris:
    
    - The city is famous for its art, music,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and unpredictable, but some potential trends that could emerge include:
    
    1. The rise of AI-driven autonomous vehicles: Autonomous vehicles powered by AI could reduce traffic accidents and pollution. They could also enhance driver safety and efficiency, leading to cost savings for governments.
    
    2. The development of AI-based language translation systems: AI-powered language translation systems could help overcome language barriers and improve communication between people from different cultures and languages.
    
    3. The use of AI in healthcare: AI could help doctors diagnose diseases, develop personalized treatments, and predict patient outcomes. AI-powered medical devices could also improve patient safety and reduce medical errors.
    
    4. The development of


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

    Your

     Job

    /

    Role

    ].

     I

    'm

     currently

     [

    Your

     Age

    ],

     but

     I

    've

     always

     been

     [

    Your

     Background

    ].

     I

    'm

     a

     [

    Your

     Profession

    /

    Activity

    ].

     Here

    's

     what

     I

     know

     about

     you

    :

     I

    'm

     passionate

     about

     [

    Your

     Passion

    ],

     and

     I

     believe

     that

     [

    Your

     Message

    /

    Value

    ].

     I

    'm

     always

     eager

     to

     learn

     and

     take

     risks

    ,

     and

     I

     enjoy

     [

    Your

     Inter

    ests

    /

    Activities

    ].

     I

     have

     a

     [

    Your

     Skill

    /

    Ability

    ]

     that

     I

    'm

     working

     on

    ,

     and

     I

    'm

     determined

     to

     [

    Your

     Goal

    ].

     Thank

     you

     for

     asking

    !

     Based

     on

     the

     information

     provided

    ,

     what

     is

     the

     profession

    /

    activities

     of

     the

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    I

     apologize

    ,

     but

     I

     cannot

     provide

     a

     factual

     statement

     about

     France

    's

     capital

     city

     without

     violating

     their

     privacy

     or

     copyright

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

     cities

     or

     their

     constit

    utions

    .

     However

    ,

     I

     can

     provide

     you

     with

     some

     factual

     information

     about

     Paris

    ,

     the

     capital

     city

     of

     France

    :
    


    Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     in

     the

     south

    -central

     part

     of

     the

     country

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     area

    ,

     with

     an

     estimated

     population

     of

     over

     

    2

    .

    1

     million

     people

     (

    as

     of

     

    2

    0

    2

    1

    ).

     The

     city

     is

     known

     for

     its

     artistic

     and

     cultural

     heritage

    ,

     including

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     full

     of

     possibilities

    !

     Here

     are

     some

     potential

     trends

     we

     can

     expect

     to

     see

    :
    


    1

    .

     More

     Human

    -like

     AI

    :

     As

     AI

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     more

     advanced

     human

    -like

     AI

     that

     exhibits

     emotions

    ,

     empathy

    ,

     and

     even

     human

    -like

     creativity

    .
    


    2

    .

     Personal

    ized

     AI

    :

     AI

     will

     become

     even

     more

     personalized

    ,

     with

     better

     understanding

     of

     individual

     needs

     and

     preferences

    .

     This

     will

     lead

     to

     more

     efficient

     and

     effective

     solutions

     to

     complex

     problems

    .
    


    3

    .

     Em

    otion

     AI

    :

     AI

     will

     become

     even

     more

     intelligent

     and

     capable

     of

     understanding

     and

     responding

     to

     emotions

    .

     This

     will

     lead

     to

     more

     personalized

     and

     effective

     emotional

     intelligence

     in

     humans

    .
    


    4

    .

     Advanced

     AI

    :

     AI

     will

    



```python
llm.shutdown()
```
