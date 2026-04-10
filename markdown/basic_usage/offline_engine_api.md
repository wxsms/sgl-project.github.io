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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.69it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.68it/s]


    2026-04-10 03:48:55,433 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 03:48:55] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:25,  2.10it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:11,  4.41it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:11,  4.41it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:03<00:11,  4.41it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:03<00:11,  4.41it/s]

    Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:03<00:11,  4.41it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:03<00:11,  4.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:05,  9.18it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:05,  9.18it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:05,  9.18it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:05,  9.18it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:05,  9.18it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:05,  9.18it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:05,  9.18it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:05,  9.18it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:05,  9.18it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 18.02it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 18.02it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 18.02it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 18.02it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 18.02it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 18.02it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 18.02it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 23.94it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 23.94it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 23.94it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 23.94it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 23.94it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 23.94it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 25.10it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 25.10it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 25.10it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 25.10it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 25.10it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 25.10it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 25.10it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 30.36it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 30.36it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 30.36it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 30.36it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 30.36it/s]

    Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 30.36it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 33.41it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 33.41it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 33.41it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 33.41it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 33.41it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 33.41it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 35.48it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 35.48it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 35.48it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 35.48it/s]

    Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 35.48it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 35.48it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 35.48it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:04<00:00, 39.76it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:04<00:00, 39.76it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 39.76it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:04<00:00, 39.76it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:04<00:00, 39.76it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:04<00:00, 39.76it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.07it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   2%|▏         | 1/58 [00:00<00:06,  9.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:06,  9.47it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:06,  9.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.37it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   9%|▊         | 5/58 [00:00<00:04, 11.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:04, 11.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:04, 11.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.03it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.03it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.06it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.74 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.19it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.27it/s] Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:01<00:01, 32.27it/s]Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.96it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.96it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.96it/s]

    Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.96it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.96it/s]Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.96it/s]Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.54it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.54it/s]Capturing num tokens (num_tokens=384 avail_mem=118.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.54it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.54it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.54it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  60%|██████    | 35/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  60%|██████    | 35/58 [00:01<00:00, 36.83it/s]

    Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  60%|██████    | 35/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  60%|██████    | 35/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=224 avail_mem=118.67 GB):  60%|██████    | 35/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=224 avail_mem=118.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.74it/s]Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.74it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.74it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.74it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.74it/s]

    Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.89it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.89it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.89it/s]Capturing num tokens (num_tokens=112 avail_mem=118.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.89it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.89it/s] Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  81%|████████  | 47/58 [00:01<00:00, 34.20it/s]Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  81%|████████  | 47/58 [00:01<00:00, 34.20it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  81%|████████  | 47/58 [00:01<00:00, 34.20it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  81%|████████  | 47/58 [00:01<00:00, 34.20it/s]

    Capturing num tokens (num_tokens=32 avail_mem=118.63 GB):  81%|████████  | 47/58 [00:01<00:00, 34.20it/s]Capturing num tokens (num_tokens=32 avail_mem=118.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.96it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.96it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.96it/s]Capturing num tokens (num_tokens=20 avail_mem=118.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.96it/s]Capturing num tokens (num_tokens=16 avail_mem=118.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.96it/s]Capturing num tokens (num_tokens=16 avail_mem=118.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.49it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.49it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.49it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.49it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:01<00:00, 29.66it/s]


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
    Generated text:  Niel, a CTO at a startup company.
    I've been working on this project for a few weeks now, and I'm facing some challenges. Please share your thoughts on the following problems, and suggest some potential solutions:
    
    Problem 1:
    I'm developing an app that requires users to enter a username and password. The user has to enter a username, but I'm not sure how to validate the length of the username. I've considered checking for the following conditions:
    1. The username must be at least 5 characters long
    2. The username must not contain any special characters or numbers
    
    I want to make sure that
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He/she is like the boss of the whole country. The president is not the only person who is in charge of the country. His/her office is the seat of power, and people in the country can easily communicate with him/her. The president is the most important person in America because of the two presidents he/she has been with the country for a long time. The two presidents are both from the same family. They are both from the same country. Both of them were born in America and came here as immigrants. Both of them became president because of their work. They are very kind and have always tried their
    ===============================
    Prompt: The capital of France is
    Generated text:  a city. The capital of Canada is a city. What is the capital of Germany? The capital of Germany is Berlin.
    Therefore, the answer is Berlin.
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of you.
    How do I start?
    It's a question that every entrepreneur, tech-savvy person and curious individual should ask themselves. But are you ready for the challenge? If you're not, then why not take the time to learn more about artificial intelligence and find out what makes it different from traditional AI?
    The field of artificial intelligence is growing rapidly and will continue to do so for years to come. However, like any new technology, AI has both advantages and disadvantages. The field is complicated, with many specialized technologies and tools that can be difficult to learn.
    The good news is that you don't have to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or experience here]. I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity here]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I love [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major cultural and economic center, hosting many world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination and a major hub for business and commerce. The city is home to many famous landmarks and attractions, including the Louvre, Notre-Dame Cathedral, and the Champs-Élysées. Paris is a vibrant and diverse city with a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced decision-making. This could lead to more personalized and context-aware AI that can better understand and respond to human emotions and behaviors.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more integrated with human intelligence, it is likely to be used in even more advanced ways, such as developing more accurate medical diagnoses and personalized treatment plans.
    
    3. Increased use of AI in manufacturing: AI is already
    


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
    Generated text:  [Name] and I'm a/an [occupation] who has always been passionate about [interest or hobby]. I'm always up for a challenge and enjoy [job title]ing up new things. I'm confident in my abilities and always strive to improve myself. I believe in the power of [skill or quality] and enjoy learning and growing every day. I'm always looking for new ways to enhance my skills and reach new heights. I'm a team player and I love to work hard with my team members to achieve our goals. I'm ready to take on new challenges and new opportunities to grow and learn. What's your name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is the largest city in France and the third-largest city in Europe. It is a historic and cultural center, renowned for its rich history, art, music, and food. It is also a major international trade and financial center. Paris is home to the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and many other famous landmarks. It is a major hub for international commerce, education, and entertainment. 
    
    Some facts about Paris include:
    
    1. It is the capital of France and is the third largest city in Europe.
    2. The Eiffel Tower is a famous landmark located in the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and highly unpredictable, but here are some potential trends that are likely to shape the industry in the coming years:
    
    1. Increased use of machine learning and deep learning: These are the key technologies that are expected to drive the growth of AI in the coming years. Machine learning algorithms are being developed to improve the accuracy and efficiency of AI systems. Deep learning is also expected to see continued growth, with more advanced models that can handle more complex and unpredictable data.
    
    2. Integration of AI into healthcare: AI is already being used in healthcare to assist with diagnosis and treatment planning. As this technology advances, we can expect to see more integration with


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

    ],

     and

     I

     am

     a

     [

    Type

    ]

     who

     has

     been

     [

    Number

     of

     Years

    ]

     years

     of

     experience

     in

     the

     [

    Industry

    ],

     and

     I

     have

     a

     strong

     [

    Professional

     Skill

    ]

     in

     [

    Skill

    ].

     I

     am

     [

    Age

    ]

     years

     old

     and

     have

     a

     passion

     for

     [

    Interest

    ].

     If

     you

    're

     ever

     in

     need

     of

     a

     professional

     help

     or

     assistance

    ,

     don

    't

     hesitate

     to

     reach

     out

     to

     me

    .

     [

    Name

    ]

     is

     ready

     to

     help

     you

    .

     [

    Name

    ]

     is

     my

     friend

     and

     I

     love

     to

     [

    Favorite

     Activity

    ].

     How

     can

     I

     assist

     you

     today

    ?

     [

    Name

    ]

     is

     a

     [

    Type

    ],

     [

    Type

    ],

     [

    Type

    ]

     who

     has

     been

     [

    Number

     of

     Years

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     significant

     cultural

    ,

     political

    ,

     and

     economic

     center

     with

     numerous

     landmarks

     and

     historical

     sites

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

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

     The

     city

     is

     also

     famous

     for

     its

     elaborate

     street

     festivals

     and

     cultural

     events

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     Paris

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

     and

     is

     known

     for

     its

     iconic

     landmarks

     and

     art

     treasures

    .

     Its

     contributions

     to

     the

     world

    's

     art

    ,

     literature

    ,

     and

     cuisine

     are

     recognized

     worldwide

    .

     The

     city

     is

     also

     home

     to

     numerous

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Mus

    ée

     Rod

    in

    .

     It

     has

     a

     diverse

     population

     of

     over

     

    2

    .

    5

     million

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     an

     exciting

     and

     rapidly

     evolving

     field

     with

     many

     potential

     paths

     and

     directions

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     are

     currently

     being

     explored

     and

     are

     likely

     to

     continue

    :
    


    1

    .

     Autonomous

     vehicles

    :

     With

     the

     development

     of

     advanced

     AI

     technologies

     such

     as

     machine

     learning

    ,

     self

    -driving

     cars

     are

     becoming

     increasingly

     common

    .

     These

     vehicles

     are

     designed

     to

     be

     able

     to

     drive

     themselves

    ,

     making

     them

     safer

     and

     more

     efficient

     than

     human

     drivers

    .
    


    2

    .

     Smart

     homes

    :

     AI

    -powered

     smart

     home

     devices

     are

     becoming

     increasingly

     popular

    ,

     enabling

     homeowners

     to

     control

     their

     homes

     from

     a

     remote

     location

    .

     These

     devices

     can

     be

     integrated

     with

     voice

     assistants

    ,

     smart

     ther

    most

    ats

    ,

     and

     other

     smart

     home

     features

    .
    


    3

    .

     AI

     for

     healthcare

    



```python
llm.shutdown()
```
