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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.77it/s]


    2026-04-15 07:01:37,194 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 07:01:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:03, 12.98it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:03, 12.98it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.98it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.98it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.98it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.98it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.98it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.98it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.98it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.02it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.02it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.02it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.02it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.02it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.02it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.02it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.02it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 26.57it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 26.57it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 26.57it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 26.57it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 26.57it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 26.57it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:03<00:00, 26.57it/s]

    Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:03<00:00, 26.57it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 33.06it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 33.06it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 33.06it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 33.06it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 33.06it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 33.06it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 33.06it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:03<00:00, 33.06it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.24it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.24it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.24it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.24it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.24it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.24it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.24it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.24it/s]

    Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.24it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.24it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 49.57it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 49.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.42 GB):   2%|▏         | 1/58 [00:00<00:05,  9.61it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.39 GB):   2%|▏         | 1/58 [00:00<00:05,  9.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.39 GB):   2%|▏         | 1/58 [00:00<00:05,  9.61it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.39 GB):   5%|▌         | 3/58 [00:00<00:04, 11.35it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.39 GB):   5%|▌         | 3/58 [00:00<00:04, 11.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.39 GB):   5%|▌         | 3/58 [00:00<00:04, 11.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.38 GB):   9%|▊         | 5/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.38 GB):   9%|▊         | 5/58 [00:00<00:03, 14.07it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 23.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.36 GB):  21%|██        | 12/58 [00:00<00:01, 23.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.36 GB):  21%|██        | 12/58 [00:00<00:01, 23.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.35 GB):  21%|██        | 12/58 [00:00<00:01, 23.29it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.35 GB):  21%|██        | 12/58 [00:00<00:01, 23.29it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.35 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.35 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.32 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=960 avail_mem=74.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.02it/s] Capturing num tokens (num_tokens=896 avail_mem=74.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=832 avail_mem=74.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.02it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=704 avail_mem=74.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=704 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.78it/s]Capturing num tokens (num_tokens=640 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.78it/s]Capturing num tokens (num_tokens=576 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.78it/s]Capturing num tokens (num_tokens=512 avail_mem=74.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.78it/s]Capturing num tokens (num_tokens=480 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:01<00:00, 36.78it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.88it/s]Capturing num tokens (num_tokens=448 avail_mem=74.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.88it/s]Capturing num tokens (num_tokens=416 avail_mem=74.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.88it/s]Capturing num tokens (num_tokens=384 avail_mem=74.31 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.88it/s]Capturing num tokens (num_tokens=352 avail_mem=74.31 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.88it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 26.43it/s]Capturing num tokens (num_tokens=320 avail_mem=74.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 26.43it/s]Capturing num tokens (num_tokens=288 avail_mem=73.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 26.43it/s]Capturing num tokens (num_tokens=256 avail_mem=73.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 26.43it/s]Capturing num tokens (num_tokens=256 avail_mem=73.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.73it/s]Capturing num tokens (num_tokens=240 avail_mem=74.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.73it/s]Capturing num tokens (num_tokens=224 avail_mem=74.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.73it/s]Capturing num tokens (num_tokens=208 avail_mem=74.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.73it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.25 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.43it/s]Capturing num tokens (num_tokens=192 avail_mem=74.03 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.43it/s]Capturing num tokens (num_tokens=176 avail_mem=74.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.43it/s]Capturing num tokens (num_tokens=160 avail_mem=74.24 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.43it/s]Capturing num tokens (num_tokens=160 avail_mem=74.24 GB):  74%|███████▍  | 43/58 [00:01<00:00, 24.83it/s]Capturing num tokens (num_tokens=144 avail_mem=74.24 GB):  74%|███████▍  | 43/58 [00:01<00:00, 24.83it/s]Capturing num tokens (num_tokens=128 avail_mem=74.24 GB):  74%|███████▍  | 43/58 [00:01<00:00, 24.83it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.23 GB):  74%|███████▍  | 43/58 [00:01<00:00, 24.83it/s]Capturing num tokens (num_tokens=112 avail_mem=74.23 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.42it/s]Capturing num tokens (num_tokens=96 avail_mem=74.22 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.42it/s] Capturing num tokens (num_tokens=80 avail_mem=74.22 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.42it/s]Capturing num tokens (num_tokens=64 avail_mem=74.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 25.42it/s]Capturing num tokens (num_tokens=64 avail_mem=74.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.47it/s]Capturing num tokens (num_tokens=48 avail_mem=74.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.47it/s]Capturing num tokens (num_tokens=32 avail_mem=74.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.47it/s]Capturing num tokens (num_tokens=28 avail_mem=74.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.47it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.18 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.47it/s]Capturing num tokens (num_tokens=24 avail_mem=74.18 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.65it/s]Capturing num tokens (num_tokens=20 avail_mem=74.17 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.65it/s]Capturing num tokens (num_tokens=16 avail_mem=74.17 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.65it/s]Capturing num tokens (num_tokens=12 avail_mem=74.16 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.65it/s]Capturing num tokens (num_tokens=8 avail_mem=74.16 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.65it/s] Capturing num tokens (num_tokens=8 avail_mem=74.16 GB):  98%|█████████▊| 57/58 [00:02<00:00, 31.23it/s]Capturing num tokens (num_tokens=4 avail_mem=74.13 GB):  98%|█████████▊| 57/58 [00:02<00:00, 31.23it/s]Capturing num tokens (num_tokens=4 avail_mem=74.13 GB): 100%|██████████| 58/58 [00:02<00:00, 26.84it/s]


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
    Generated text:  Tia and I’m a 13-year-old boy from Portland, Oregon. I’m an Eagle Scout (grades 1-6). What is your name and where are you from? I'm 13 and I'm from Portland, Oregon. I'm an Eagle Scout (grades 1-6). How did you become an Eagle Scout?
    
    You are welcome, Tia. You’re an Eagle Scout (grades 1-6). When I started my journey with Eagle Scout, it was because my family had joined the Eagle Scouts in the school. I remember it was my father who made me learn everything he could about
    ===============================
    Prompt: The president of the United States is
    Generated text:  36 years older than the president of Brazil, and the president of Brazil is 3/4 times older than the president of China. If the president of the United States is currently 44 years old, how old would the president of the United States be in 10 years?
    To find out how old the president of the United States would be in 10 years, we need to follow a few steps to determine the current age of the president of Brazil and then the president of China.
    
    First, we know that the president of the United States is currently 44 years old. According to the problem, the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris  
    B) Lyon  
    C) Marseille  
    D) Bordeaux
    
    To determine the capital of France, let's recall the capital of France, which is Paris.
    
    So, the correct answer is:
    
    A) Paris
    
    We can confirm this by checking the options provided:
    B) Lyon - This is a city in France, but it is not the capital.
    C) Marseille - This is a city in France, but it is not the capital.
    D) Bordeaux - This is a city in France, but it is not the capital.
    
    Therefore, the capital of France is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. But we need to be careful about what we are going to do with the data and the tools and the people we are going to trust.
    Those are the words of an executive at the World Economic Forum in Davos. The purpose of this post is to write about how governments and individuals can and should use data to drive innovation.
    The world's most powerful companies, including Apple, Google and IBM, have all committed to sharing large volumes of data with others. The resulting data democratization has led to a great deal of innovation.
    The big problems, however, are the following:
    1) Data is a valuable resource, but it


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


    Generated text:  Paris, also known as "La Ville de Paris". It is the largest city in France and the third largest in the world, with a population of over 2. 5 million people. Paris is known for its rich history, art, and culture, and is a popular tourist destination. It is also home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a cultural hub and a major economic center in France, and is a major transportation hub for Europe. It is also home to many important institutions such as the French Academy of Sciences and the French Parliament. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency, reduced costs, and improved productivity.
    
    2. Enhanced human-AI collaboration: As AI becomes more advanced, we can expect to see more collaboration between humans and AI. This could lead to more effective problem-solving, improved decision-making, and enhanced creativity.
    
    3. AI-driven healthcare: AI is already
    


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
    Generated text:  Alex, and I'm a professional at heart. I'm a marketing strategist with a background in advertising and brand development. I have a knack for understanding consumer behavior and using data to drive successful marketing campaigns. I thrive on creating engaging content and using social media to build a loyal following. I'm always looking for ways to improve my skills and stay up-to-date with the latest marketing trends. I'm confident in my ability to make a positive impact on the world through my work. Thank you. Here's a neutral self-introduction for Alex:
    
    ---
    
    Hello, my name is Alex, and I'm a marketing strategist with a background in advertising
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its famous landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Museum. 
    
    France's capital city is also home to some of the world's most prestigious institutions, including the National Museum of Modern Art (Musée d'Orsay) and the Musée Rodin. Paris is also a popular tourist destination for its delicious cuisine, beautiful architecture, and cultural events. 
    
    It's worth noting that Paris is often referred to as the "City of Light" due to its many iconic illuminated areas and neon signs, while its nickname "City of Love"
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid advancements in technology, increased use of AI in various industries, and a greater focus on ethical considerations and responsible AI development. Here are some possible trends in AI:
    
    1. Increased integration with natural language processing (NLP): AI will continue to integrate with NLP, allowing it to process and understand human language more effectively. This will enable AI to better understand and interpret complex natural language and to generate human-like responses.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be increased concerns about privacy and security. This will likely lead to more advanced privacy and security measures and greater emphasis on


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

    name

    ].

     I

    ’m

     a

     [

    age

    ]

     year

     old

     [

    gender

    ]

     with

     [

    birth

    place

    ]

     [

    city

    ].

     I

     have

     lived

     in

     [

    city

    ]

     since

     [

    year

    ]

     [

    month

    ].

     I

     was

     raised

     [

    occupation

    ].

     I

     enjoy

     [

    way

     to

     unwind

    ].

     I

     love

     [

    food

    ],

     [

    h

    obby

    ],

     [

    activity

    ],

     [

    sport

    ],

     and

     [

    anything

     else

     I

     enjoy

    ].

     I

     want

     to

     live

     in

     [

    city

    ]

     for

     [

    reason

    ].

     I

     believe

     that

     [

    reason

    ]

     makes

     my

     life

     more

     enjoyable

     for

     [

    person

    ].

     How

     would

     you

     describe

     yourself

     to

     someone

     who

     knows

     you

    ?

     Hello

    ,

     my

     name

     is

     [

    name

    ].

     I

    ’m

     a

     [

    age

    ]

     year

     old

     [

    gender

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    A

     concise

     factual

     statement

     about

     France

    ’s

     capital

     city

     is

     that

     it

     is

     the

     capital

     city

     of

     France

     and

     one

     of

     the

     world

    ’s

     most

     important

     cities

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     diverse

     culture

    ,

     making

     it

     a

     popular

     tourist

     destination

     and

     a

     center

     of

     French

     society

    .

     It

     is

     also

     a

     major

     financial

     hub

    ,

     with

     the

     headquarters

     of

     some

     of

     the

     world

    's

     biggest

     corporations

     located

     in

     the

     city

    .

     The

     city

     is

     home

     to

     many

     iconic

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    ,

     and

     is

     an

     important

     center

     of

     politics

    ,

     culture

    ,

     and

     arts

     in

     France

     and

     the

     world

    .

     The

     city

     is

     also

     home

     to

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     an

     increasing

     focus

     on

     creating

     more

     advanced

     and

     sophisticated

     AI

     systems

     that

     can

     perform

     a

     wider

     range

     of

     tasks

     with

     increasing

     accuracy

     and

     efficiency

    .

     Here

     are

     some

     possible

     trends

     that

     may

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     and

     safety

     concerns

    :

     As

     AI

     systems

     become

     more

     complex

     and

     sophisticated

    ,

     there

     is

     a

     growing

     concern

     about

     their

     potential

     to

     cause

     harm

     or

     unintended

     consequences

    .

     The

     focus

     may

     shift

     from

     purely

     technical

     improvement

     to

     a

     more

     ethical

     and

     safety

    -focused

     approach

    .

     This

     may

     involve

     developing

     frameworks

     that

     require

     AI

     systems

     to

     be

     transparent

    ,

     accountable

    ,

     and

     responsible

     for

     their

     actions

    .
    


    2

    .

     Development

     of

     new

     machine

     learning

     algorithms

    :

     As

     AI

     systems

     become

    



```python
llm.shutdown()
```
