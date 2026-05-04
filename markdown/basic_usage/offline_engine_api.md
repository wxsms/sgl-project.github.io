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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.81it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.80it/s]


    2026-05-04 15:08:35,619 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 15:08:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.68it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.68it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:12,  3.68it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:12,  3.68it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:12,  3.68it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:12,  3.68it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:12,  3.68it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.68it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.68it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.71it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.71it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.71it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.71it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.71it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.71it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.71it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.71it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 12.11it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 12.11it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 12.11it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 12.11it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 12.11it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 12.11it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 12.11it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 12.11it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 17.40it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 17.40it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 17.40it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 17.40it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 17.40it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 17.40it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 17.40it/s]

    Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:05<00:01, 17.40it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:05<00:01, 17.40it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 24.24it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 24.24it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 24.24it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 24.24it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 24.24it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 24.24it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 24.24it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 24.24it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.62it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.62it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.62it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.62it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.62it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.62it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.62it/s]

    Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.62it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.62it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.62it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.06 GB):   3%|▎         | 2/58 [00:00<00:03, 17.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.06 GB):   3%|▎         | 2/58 [00:00<00:03, 17.08it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.06 GB):   3%|▎         | 2/58 [00:00<00:03, 17.08it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.06 GB):   7%|▋         | 4/58 [00:00<00:03, 17.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.05 GB):   7%|▋         | 4/58 [00:00<00:03, 17.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.05 GB):   7%|▋         | 4/58 [00:00<00:03, 17.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.04 GB):   7%|▋         | 4/58 [00:00<00:03, 17.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.04 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.04 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.03 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.03 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.92it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.03 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.03 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.02 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.02 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.02 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.01 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.01 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.37it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.01 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.01 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.37it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=72.00 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.00 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.98 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.89it/s]Capturing num tokens (num_tokens=960 avail_mem=71.99 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.89it/s] Capturing num tokens (num_tokens=896 avail_mem=71.99 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.89it/s]Capturing num tokens (num_tokens=896 avail_mem=71.99 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.82it/s]Capturing num tokens (num_tokens=832 avail_mem=71.99 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.82it/s]

    Capturing num tokens (num_tokens=768 avail_mem=71.98 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.82it/s]Capturing num tokens (num_tokens=704 avail_mem=71.98 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.82it/s]Capturing num tokens (num_tokens=640 avail_mem=71.98 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.82it/s]Capturing num tokens (num_tokens=640 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:01<00:01, 24.77it/s]Capturing num tokens (num_tokens=576 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:01<00:01, 24.77it/s]

    Capturing num tokens (num_tokens=512 avail_mem=71.96 GB):  47%|████▋     | 27/58 [00:01<00:01, 24.77it/s]Capturing num tokens (num_tokens=480 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:01<00:01, 24.77it/s]Capturing num tokens (num_tokens=480 avail_mem=71.98 GB):  52%|█████▏    | 30/58 [00:01<00:01, 17.78it/s]Capturing num tokens (num_tokens=448 avail_mem=71.98 GB):  52%|█████▏    | 30/58 [00:01<00:01, 17.78it/s]Capturing num tokens (num_tokens=416 avail_mem=71.97 GB):  52%|█████▏    | 30/58 [00:01<00:01, 17.78it/s]

    Capturing num tokens (num_tokens=384 avail_mem=71.97 GB):  52%|█████▏    | 30/58 [00:01<00:01, 17.78it/s]Capturing num tokens (num_tokens=384 avail_mem=71.97 GB):  57%|█████▋    | 33/58 [00:01<00:01, 18.47it/s]Capturing num tokens (num_tokens=352 avail_mem=71.97 GB):  57%|█████▋    | 33/58 [00:01<00:01, 18.47it/s]Capturing num tokens (num_tokens=320 avail_mem=71.96 GB):  57%|█████▋    | 33/58 [00:01<00:01, 18.47it/s]Capturing num tokens (num_tokens=288 avail_mem=71.96 GB):  57%|█████▋    | 33/58 [00:01<00:01, 18.47it/s]Capturing num tokens (num_tokens=288 avail_mem=71.96 GB):  62%|██████▏   | 36/58 [00:01<00:01, 19.79it/s]Capturing num tokens (num_tokens=256 avail_mem=71.95 GB):  62%|██████▏   | 36/58 [00:01<00:01, 19.79it/s]Capturing num tokens (num_tokens=240 avail_mem=71.95 GB):  62%|██████▏   | 36/58 [00:01<00:01, 19.79it/s]

    Capturing num tokens (num_tokens=224 avail_mem=71.95 GB):  62%|██████▏   | 36/58 [00:01<00:01, 19.79it/s]Capturing num tokens (num_tokens=224 avail_mem=71.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 21.89it/s]Capturing num tokens (num_tokens=208 avail_mem=71.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 21.89it/s]Capturing num tokens (num_tokens=192 avail_mem=71.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 21.89it/s]Capturing num tokens (num_tokens=176 avail_mem=71.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 21.89it/s]Capturing num tokens (num_tokens=176 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.74it/s]Capturing num tokens (num_tokens=160 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.74it/s]Capturing num tokens (num_tokens=144 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.74it/s]Capturing num tokens (num_tokens=128 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.74it/s]

    Capturing num tokens (num_tokens=112 avail_mem=71.92 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.74it/s]Capturing num tokens (num_tokens=112 avail_mem=71.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.85it/s]Capturing num tokens (num_tokens=96 avail_mem=71.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.85it/s] Capturing num tokens (num_tokens=80 avail_mem=71.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.85it/s]Capturing num tokens (num_tokens=64 avail_mem=71.91 GB):  79%|███████▉  | 46/58 [00:02<00:00, 25.85it/s]Capturing num tokens (num_tokens=48 avail_mem=71.91 GB):  79%|███████▉  | 46/58 [00:02<00:00, 25.85it/s]Capturing num tokens (num_tokens=48 avail_mem=71.91 GB):  86%|████████▌ | 50/58 [00:02<00:00, 27.58it/s]Capturing num tokens (num_tokens=32 avail_mem=71.90 GB):  86%|████████▌ | 50/58 [00:02<00:00, 27.58it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.90 GB):  86%|████████▌ | 50/58 [00:02<00:00, 27.58it/s]Capturing num tokens (num_tokens=24 avail_mem=71.90 GB):  86%|████████▌ | 50/58 [00:02<00:00, 27.58it/s]Capturing num tokens (num_tokens=24 avail_mem=71.90 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.10it/s]Capturing num tokens (num_tokens=20 avail_mem=71.90 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.10it/s]Capturing num tokens (num_tokens=16 avail_mem=71.89 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.10it/s]Capturing num tokens (num_tokens=12 avail_mem=71.89 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.10it/s]Capturing num tokens (num_tokens=8 avail_mem=71.89 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.10it/s] Capturing num tokens (num_tokens=8 avail_mem=71.89 GB):  98%|█████████▊| 57/58 [00:02<00:00, 30.33it/s]Capturing num tokens (num_tokens=4 avail_mem=71.88 GB):  98%|█████████▊| 57/58 [00:02<00:00, 30.33it/s]Capturing num tokens (num_tokens=4 avail_mem=71.88 GB): 100%|██████████| 58/58 [00:02<00:00, 24.88it/s]


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
    Generated text:  Tania and I am a senior at my school who is going to the United States for summer. I hope to see you on June 1st. I'm writing to tell you about my family and what I have been up to. I have two brothers, and two sisters. My family is very close and we all love each other very much. My mother and father are lawyers and my father is also a firefighter. My mother's name is Jessica and she is from Ireland. She loves to cook food and also likes to sing in a band. My brother, Joseph is a junior at Monroe High School. My sister is in high
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by a vice president who is the second in command of the presidency. 
    
    In the United States, what is the term of office for the vice president? 
    A. 1 year 
    B. 2 years 
    C. 3 years 
    D. 4 years
    To determine the term of office for the vice president of the United States, let's analyze the information provided and the options given.
    
    1. The president is the chief executive of the United States.
    2. The vice president, in turn, is the second in command of the presidency.
    
    Given that the vice president is second in command, it implies that he
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the center of a rectangular plot of land, measuring 120 meters by 80 meters. It is observed that the road running from one corner of the plot to the opposite corner is 30 meters wide. The capital city is surrounded by a fence. Calculate the total length of the fence needed to enclose the entire rectangular plot, including the road. Assume the plot is to be divided into two rectangular sections by the road running diagonally from one corner of the plot to the opposite corner. To determine the total length of the fence needed to enclose the entire rectangular plot, including the road, we first need
    ===============================
    Prompt: The future of AI is
    Generated text:  moving forward in the digital age. With the rise of AI-powered technologies, the personalization of data and the development of more complex algorithms have transformed the way we interact with the world. These changes have raised significant concerns, including the possibility of AI exacerbating inequalities and biases in society. The report, "Beyond Bias: The Future of AI in the Digital Age," has been created to explore the future of AI and its impact on society. It explores the rise of AI-powered technologies, the ways in which they are used to personalize data, and the challenges they present. The report also examines the ethical implications of AI and the role of human


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? I'm a [insert a few details about your personality, skills, and accomplishments]. And what's your background? I'm [insert a few details about your education, work experience, or other relevant information]. And what's your favorite hobby or activity? I'm [insert a few details about your hobbies or interests]. And what's your favorite book or movie? I'm [insert a few details about your favorite books or movies]. And what
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination and a major hub for business and commerce in Europe. The city is home to many famous French artists, writers, and musicians, and is known for its rich history and cultural heritage. Paris is a vibrant and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions between humans and machines.
    
    2. Enhanced privacy and security: As AI becomes more prevalent, there will be increased concerns about privacy and security. There will be a need for more robust privacy protections and measures to ensure that AI systems are not used to invade personal data.
    
    3. Greater focus on ethical considerations: As AI becomes more advanced, there
    


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
    Generated text:  John Smith. I'm a 35-year-old software engineer with a passion for programming, design, and problem-solving. I'm always looking for ways to improve my skills and stay up-to-date with the latest technologies. I enjoy spending time with friends and taking care of my family. I'm looking forward to meeting you. Have a great day! [Optional: add your role in the software development team or any other relevant details to make the introduction more engaging.] [Remember, the key is to keep the introduction neutral and professional, without being overly formal or biased.] John Smith, Software Engineer at XYZ Corporation. 🌟 My
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "la Ville de Paris", which was founded in the 13th century and is the most populous city in the European Union and the world’s largest city by population. It is also the oldest continuously inhabited city in the world and the birthplace of many French national symbols and institutions. Paris is known for its rich history, architecture, and cultural heritage, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also home to the French National Library, the Arc de Triomphe, and the Louvre Museum. The city is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and constantly evolving, with many potential directions for development. Here are some of the most likely trends that could shape AI in the coming years:
    
    1. Self-driving cars: As technology continues to improve, self-driving cars could become more common in the future. These vehicles would be able to recognize and avoid obstacles and navigate through cities with the help of sensors and cameras.
    
    2. Personalized medicine: AI could be used to analyze medical data to help doctors make more accurate diagnoses and treatments. This could lead to more effective and personalized healthcare, with fewer side effects.
    
    3. Autonomous systems: AI could be used to create autonomous vehicles that can


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

    /an

     [

    age

    ]

     year

     old

     girl

    .

     I

    'm

     an

     [

    occupation

    ]

     who

     has

     a

     passion

     for

     [

    interest

    ].

     I

     like

     to

     [

    make

     a

     statement

     about

     my

     hobbies

     or

     interests

    ].

     I

    'm

     confident

     and

     creative

    ,

     always

     striving

     to

     learn

     and

     improve

    .

     I

    'm

     a

     [

    level

    ]

     learner

     and

     [

    name

     of

     book

    /

    TV

     show

    /

    website

     I

     enjoy

     reading

    ,

     watching

    ,

     or

     following

    ].

     I

     love

     to

     [

    describe

     a

     hobby

     or

     activity

     I

     enjoy

     doing

    ].

     If

     you

    're

     interested

     in

     learning

     more

     about

     me

    ,

     feel

     free

     to

     ask

     me

     anything

    .

     I

    'm

     [

    gender

    ]

     and

     I

    'm

     a

    /an

     [

    race

    /class

    ification

    ]

     person

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

     and

     world

    -ren

    owned

     museums

     like

     the

     Lou

    vre

     and

     Mus

    ée

     d

    '

    Or

    say

    .

     Additionally

    ,

     Paris

     is

     the

     birth

    place

     of

     the

     French

     Revolution

     and

     is

     a

     popular

     tourist

     destination

     for

     its

     rich

     cultural

     and

     historical

     sites

    ,

     including

     the

     Lou

    vre

     and

     Notre

    -D

    ame

     Cathedral

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     dominated

     by

     a

     number

     of

     key

     trends

     that

     will

     shape

     the

     way

     that

     humans

     interact

     with

     and

     create

     AI

     systems

    .

     Some

     of

     these

     trends

     include

    :
    


    1

    .

     More

     advanced

     AI

     will

     be

     able

     to

     learn

     from

     more

     complex

     and

     varied

     data

    ,

     allowing

     for

     more

     sophisticated

     and

     nuanced

     decision

    -making

    .
    


    2

    .

     AI

     will

     become

     more

     integrated

     into

     society

    ,

     with

     more

     people

     working

     in

     roles

     that

     require

     advanced

     AI

     skills

    .
    


    3

    .

     AI

     will

     become

     more

     accessible

     to

     a

     wider

     range

     of

     people

    ,

     with

     more

     people

     having

     access

     to

     affordable

     and

     easily

    -

    learn

    ed

     AI

     systems

    .
    


    4

    .

     AI

     will

     become

     more

     autonomous

    ,

     with

     machines

     being

     able

     to

     make

     decisions

     on

     their

     own

     without

     human

     intervention

    



```python
llm.shutdown()
```
