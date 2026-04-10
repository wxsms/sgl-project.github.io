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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.94it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.93it/s]


    2026-04-10 03:13:42,709 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 03:13:42] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:02<00:11,  4.35it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:02<00:11,  4.35it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:02<00:11,  4.35it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:02<00:11,  4.35it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:03<00:11,  4.35it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:03<00:11,  4.35it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:03<00:11,  4.35it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:04,  9.12it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:04,  9.12it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:04,  9.12it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:04,  9.12it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:04,  9.12it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:03<00:04,  9.12it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:03<00:04,  9.12it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:03<00:04,  9.12it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 15.63it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 15.63it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 15.63it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 15.63it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 15.63it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 15.63it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 15.63it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 15.63it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 22.37it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 22.37it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 22.37it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 22.37it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 22.37it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 22.37it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:03<00:01, 22.37it/s]

    Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:03<00:01, 22.37it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 32.32it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 32.32it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 32.32it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 32.32it/s]

    Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 32.32it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 32.32it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 32.32it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 36.47it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 36.47it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 36.47it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 36.47it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 36.47it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 36.47it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 36.47it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 41.34it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 41.34it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 41.34it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 41.34it/s]

    Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 41.34it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 41.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.96 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.96 GB):   3%|▎         | 2/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.95 GB):   3%|▎         | 2/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.95 GB):   3%|▎         | 2/58 [00:00<00:03, 15.42it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.95 GB):   3%|▎         | 2/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.95 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.95 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.94 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.62it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.90 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.90 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.87 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.63it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.86 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.63it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.86 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.63it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.86 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.85 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.85 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.77it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=118.84 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.84 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.84 GB):  31%|███       | 18/58 [00:00<00:01, 25.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.84 GB):  31%|███       | 18/58 [00:00<00:01, 25.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.83 GB):  31%|███       | 18/58 [00:00<00:01, 25.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.81 GB):  31%|███       | 18/58 [00:00<00:01, 25.82it/s]Capturing num tokens (num_tokens=960 avail_mem=118.83 GB):  31%|███       | 18/58 [00:00<00:01, 25.82it/s] Capturing num tokens (num_tokens=960 avail_mem=118.83 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=896 avail_mem=118.83 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=832 avail_mem=118.82 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.12it/s]

    Capturing num tokens (num_tokens=768 avail_mem=118.82 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=704 avail_mem=118.82 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=640 avail_mem=118.81 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.12it/s]Capturing num tokens (num_tokens=640 avail_mem=118.81 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.17it/s]Capturing num tokens (num_tokens=576 avail_mem=118.81 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.17it/s]Capturing num tokens (num_tokens=512 avail_mem=118.80 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.17it/s]Capturing num tokens (num_tokens=480 avail_mem=118.82 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.17it/s]Capturing num tokens (num_tokens=448 avail_mem=118.81 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.17it/s]Capturing num tokens (num_tokens=416 avail_mem=118.81 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.17it/s]Capturing num tokens (num_tokens=416 avail_mem=118.81 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.22it/s]Capturing num tokens (num_tokens=384 avail_mem=118.81 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.22it/s]

    Capturing num tokens (num_tokens=352 avail_mem=118.80 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.22it/s]Capturing num tokens (num_tokens=320 avail_mem=118.80 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.22it/s]Capturing num tokens (num_tokens=288 avail_mem=118.80 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.22it/s]Capturing num tokens (num_tokens=288 avail_mem=118.80 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.55it/s]Capturing num tokens (num_tokens=256 avail_mem=118.79 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.55it/s]Capturing num tokens (num_tokens=240 avail_mem=118.79 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.55it/s]Capturing num tokens (num_tokens=224 avail_mem=118.74 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.55it/s]Capturing num tokens (num_tokens=208 avail_mem=118.73 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.55it/s]Capturing num tokens (num_tokens=192 avail_mem=118.71 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.55it/s]Capturing num tokens (num_tokens=192 avail_mem=118.71 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=176 avail_mem=118.71 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]

    Capturing num tokens (num_tokens=160 avail_mem=118.71 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=144 avail_mem=118.70 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=128 avail_mem=118.70 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=112 avail_mem=118.70 GB):  71%|███████   | 41/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=112 avail_mem=118.70 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.18it/s]Capturing num tokens (num_tokens=96 avail_mem=118.69 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.18it/s] Capturing num tokens (num_tokens=80 avail_mem=118.69 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.18it/s]Capturing num tokens (num_tokens=64 avail_mem=118.69 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.18it/s]Capturing num tokens (num_tokens=48 avail_mem=118.69 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.18it/s]

    Capturing num tokens (num_tokens=32 avail_mem=118.68 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.18it/s]Capturing num tokens (num_tokens=32 avail_mem=118.68 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.54it/s]Capturing num tokens (num_tokens=28 avail_mem=118.66 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.54it/s]Capturing num tokens (num_tokens=24 avail_mem=118.65 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.54it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.54it/s]Capturing num tokens (num_tokens=16 avail_mem=118.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.54it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.54it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.82it/s] Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.82it/s]

    Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:01<00:00, 32.43it/s]


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
    Generated text:  Jennifer Liu and I am a second-year Medical Student at the University of Pittsburgh. I am in the Department of Surgery. I have had the privilege of attending a number of medical school programs and learning a lot of material. After completing my medical school studies, I am looking to enter the world of public health and want to work in a specific area of public health.
    
    I have a passion for improving the health of the United States. I am also very interested in working with vulnerable populations and have been involved in volunteering in public health, with a particular focus on diabetes management and maternal health.
    
    I am very excited to be a part of the medical
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected every four years. If the president is elected today, in how many years will the next president be born, assuming that the current president was born in the year 2000?
    To determine in how many years the next president will be born, we need to understand the cycle of presidential elections and the birth years of the current president.
    
    1. The president is elected every four years.
    2. The current president was born in 2000.
    3. The president will be born in the year 2004, which is 4 years after 2000.
    4. The president will be
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is located on the banks of the Seine River and is a city of many landmarks including the Eiffel Tower. The Eiffel Tower is a famous landmark in Paris that has been standing for over a century. The tower has a history of over 200 years and has stood since 1889. The tower was designed by Gustav Klimt and was created by the famous French architect Édouard Despley. The tower was originally called the Louvre, and was first completed in 1793 and was built on the site of a 16th-century fortress
    ===============================
    Prompt: The future of AI is
    Generated text:  coming. For those who don’t know, AI is a type of intelligent software that can simulate intelligence in any kind of machine or device. AI is expected to bring many benefits to the future, such as improving decision making and inventing new technologies.
    There are many areas of AI that are still being researched and developed. For example, AI is being used in fields such as healthcare and education. AI can also be used to improve the speed of healthcare delivery and improve the efficiency of education. There are also potential benefits to AI that are not immediately obvious.
    AI is a fascinating and exciting field. There is no doubt that it will play a


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of French Revolution and Napoleon Bonaparte, and its influence on modern French culture and politics. It is also a popular tourist destination, attracting millions of visitors each year. Paris is a vibrant and diverse city with a rich cultural heritage that continues to influence the country and the world. The city is also known for its cuisine, including its famous French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be a greater need for privacy and security measures to protect the data and personal information that is generated and processed by AI systems. This could lead to more stringent
    


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
    Generated text:  [Your Name]. I'm a [Your Profession] with [Your Interests, hobbies, and experiences] and have a passion for [Your Interest/Interest]. I’m always ready to share my knowledge and experiences with anyone who’s willing to listen. What’s your favorite hobby? What’s your dream vacation? What’s your favorite book? What’s your favorite movie? What’s your favorite food? What’s your favorite color? What’s your favorite place to live? What’s your favorite mode of transportation? What’s your favorite sports? What’s your favorite music? What’s your favorite type of music? What’s your favorite tattoo
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The French government, French media, and the French public all come from Paris. Paris is a city with a unique blend of history and culture. The city is also renowned for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a vibrant, modern city that has become a global center of culture and business. French cuisine, music, and fashion are all heavily influenced by Paris culture. 
    
    When you visit Paris, make sure to try the famous French dishes, such as coq au vin, escargot, and croissant. The city is also known
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking very bright, with many trends set to shape the field in the years ahead. Here are some possible future trends in AI:
    
    1. Increased relevance and applications in other industries: As more and more companies and governments invest in AI, it is likely that the technology will become more relevant in other industries as well. For example, AI could be used to improve healthcare by predicting disease outbreaks and developing personalized treatment plans. AI could also be used to enhance transportation, such as improving traffic flow and reducing traffic congestion.
    
    2. Greater automation: While AI is becoming increasingly sophisticated, there will still be some jobs that need to be done by humans.


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

     Jane

    .

     I

    'm

     a

     high

     school

     student

     from

     San

     Francisco

    .

     I

     love

     to

     read

     and

     write

     stories

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     express

     my

     creativity

    .

     I

    'm

     passionate

     about

     technology

     and

     I

    'm

     always

     looking

     for

     ways

     to

     make

     my

     community

     more

     tech

    -s

    av

    vy

    .

     I

    'm

     a

     bit

     of

     a

     social

     media

     junk

    ie

    ,

     and

     I

     love

     to

     follow

     my

     favorite

     accounts

     and

     participate

     in

     community

     events

    .

     I

    'm

     excited

     to

     be

     part

     of

     a

     group

     that

     values

     diversity

     and

     inclusion

    .

     Thanks

     for

     having

     me

    !

     What

     kind

     of

     activities

     do

     you

     participate

     in

    ,

     and

     what

     does

     your

     favorite

     hobby

     involve

    ?

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

     personal

     activities

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    I

    'm

     sorry

    ,

     but

     I

     cannot

     provide

     an

     answer

     to

     that

     question

     as

     there

     is

     no

     factual

     statement

     about

     Paris

     being

     the

     capital

     of

     France

    .

     Paris

     is

     the

     capital

     city

     of

     France

     and

     it

     is

     home

     to

     the

     country

    's

     government

     and

     many

     of

     its

     major

     institutions

    .

     The

     city

     is

     also

     home

     to

     many

     important

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     However

    ,

     I

    'm

     not

     able

     to

     provide

     a

     concise

     factual

     statement

     about

     Paris

    's

     other

     aspects

    ,

     such

     as

     its

     cuisine

     or

     its

     fashion

     industry

    .

     Is

     there

     anything

     else

     I

     can

     help

     you

     with

    ?

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     a

     blend

     of

     both

     traditional

     and

     emerging

     technologies

    .

     Some

     possible

     trends

     include

    :
    


    1

    .

     Increased

     automation

     and

     robotics

    :

     AI

     will

     continue

     to

     evolve

    ,

     leading

     to

     more

     sophisticated

     robots

     and

     machines

     that

     can

     perform

     tasks

     with

     greater

     efficiency

     and

     accuracy

     than

     humans

    .

     This

     could

     lead

     to

     significant

     job

     loss

    ,

     but

     also

     create

     new

     opportunities

     for

     workers

     to

     transition

     to

     new

     roles

    .
    


    2

    .

     Aug

    mented

     and

     virtual

     reality

    :

     AI

     will

     continue

     to

     improve

     the

     quality

     of

     virtual

     and

     augmented

     reality

     experiences

    ,

     making

     them

     more

     immersive

     and

     engaging

    .

     This

     could

     have

     a

     significant

     impact

     on

     the

     way

     we

     consume

     entertainment

     and

     educational

     content

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     will

     be

     used

     in

     healthcare

     to

    



```python
llm.shutdown()
```
