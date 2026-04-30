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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.40it/s]


    2026-04-30 18:32:43,915 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 18:32:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:19,  2.59it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.60it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:05,  7.08it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:05,  7.08it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:05,  7.08it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:05,  7.08it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:05<00:05,  7.08it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:05<00:05,  7.08it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:03, 10.86it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:03, 10.86it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:03, 10.86it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:03, 10.86it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:03, 10.86it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:03, 10.86it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 15.08it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 15.08it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 15.08it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 15.08it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 15.08it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 15.08it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 19.83it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 25.62it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 25.62it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 25.62it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 25.62it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 25.62it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 25.62it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 25.62it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 25.62it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 25.62it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 35.13it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 35.13it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 35.13it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 35.13it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 35.13it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 35.13it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 35.13it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 35.13it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 35.13it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 43.95it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 43.95it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 43.95it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 43.95it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 43.95it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 43.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.26 GB):   3%|▎         | 2/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.26 GB):   3%|▎         | 2/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.25 GB):   3%|▎         | 2/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.25 GB):   3%|▎         | 2/58 [00:00<00:02, 19.19it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.24 GB):   9%|▊         | 5/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.69 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.20 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.81it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=70.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.18it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.19 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.19 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.19 GB):  21%|██        | 12/58 [00:00<00:02, 16.90it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.18 GB):  21%|██        | 12/58 [00:00<00:02, 16.90it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.18 GB):  21%|██        | 12/58 [00:00<00:02, 16.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.18 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.81 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.17 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.17 GB):  28%|██▊       | 16/58 [00:00<00:02, 17.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.83 GB):  28%|██▊       | 16/58 [00:00<00:02, 17.22it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=71.16 GB):  28%|██▊       | 16/58 [00:00<00:02, 17.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.16 GB):  31%|███       | 18/58 [00:01<00:02, 17.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.16 GB):  31%|███       | 18/58 [00:01<00:02, 17.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.88 GB):  31%|███       | 18/58 [00:01<00:02, 17.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.14 GB):  31%|███       | 18/58 [00:01<00:02, 17.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.14 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.85it/s]Capturing num tokens (num_tokens=960 avail_mem=71.15 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.85it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=70.91 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.85it/s]Capturing num tokens (num_tokens=832 avail_mem=71.14 GB):  36%|███▌      | 21/58 [00:01<00:01, 18.85it/s]Capturing num tokens (num_tokens=832 avail_mem=71.14 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.57it/s]Capturing num tokens (num_tokens=768 avail_mem=71.13 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.57it/s]Capturing num tokens (num_tokens=704 avail_mem=71.12 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.57it/s]Capturing num tokens (num_tokens=640 avail_mem=70.92 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.57it/s]Capturing num tokens (num_tokens=640 avail_mem=70.92 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.21it/s]Capturing num tokens (num_tokens=576 avail_mem=71.12 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.21it/s]

    Capturing num tokens (num_tokens=512 avail_mem=71.09 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.21it/s]Capturing num tokens (num_tokens=480 avail_mem=71.11 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.21it/s]Capturing num tokens (num_tokens=480 avail_mem=71.11 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.68it/s]Capturing num tokens (num_tokens=448 avail_mem=71.10 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.68it/s]Capturing num tokens (num_tokens=416 avail_mem=70.97 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.68it/s]Capturing num tokens (num_tokens=384 avail_mem=70.97 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.68it/s]Capturing num tokens (num_tokens=352 avail_mem=71.00 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.68it/s]Capturing num tokens (num_tokens=352 avail_mem=71.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 26.23it/s]Capturing num tokens (num_tokens=320 avail_mem=71.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 26.23it/s]

    Capturing num tokens (num_tokens=288 avail_mem=71.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 26.23it/s]Capturing num tokens (num_tokens=256 avail_mem=71.03 GB):  59%|█████▊    | 34/58 [00:01<00:00, 26.23it/s]Capturing num tokens (num_tokens=240 avail_mem=70.99 GB):  59%|█████▊    | 34/58 [00:01<00:00, 26.23it/s]Capturing num tokens (num_tokens=240 avail_mem=70.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.50it/s]Capturing num tokens (num_tokens=224 avail_mem=71.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.50it/s]Capturing num tokens (num_tokens=208 avail_mem=70.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.50it/s]Capturing num tokens (num_tokens=192 avail_mem=70.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.50it/s]Capturing num tokens (num_tokens=176 avail_mem=70.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.50it/s]Capturing num tokens (num_tokens=176 avail_mem=70.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.90it/s]Capturing num tokens (num_tokens=160 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.90it/s]

    Capturing num tokens (num_tokens=144 avail_mem=70.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.90it/s]Capturing num tokens (num_tokens=128 avail_mem=70.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.90it/s]Capturing num tokens (num_tokens=112 avail_mem=70.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.90it/s]Capturing num tokens (num_tokens=112 avail_mem=70.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.82it/s]Capturing num tokens (num_tokens=96 avail_mem=70.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.82it/s] Capturing num tokens (num_tokens=80 avail_mem=70.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.82it/s]Capturing num tokens (num_tokens=64 avail_mem=70.98 GB):  79%|███████▉  | 46/58 [00:02<00:00, 32.82it/s]Capturing num tokens (num_tokens=48 avail_mem=70.97 GB):  79%|███████▉  | 46/58 [00:02<00:00, 32.82it/s]Capturing num tokens (num_tokens=32 avail_mem=70.98 GB):  79%|███████▉  | 46/58 [00:02<00:00, 32.82it/s]

    Capturing num tokens (num_tokens=32 avail_mem=70.98 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.23it/s]Capturing num tokens (num_tokens=28 avail_mem=70.96 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.23it/s]Capturing num tokens (num_tokens=24 avail_mem=70.96 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.23it/s]Capturing num tokens (num_tokens=20 avail_mem=70.95 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.23it/s]Capturing num tokens (num_tokens=16 avail_mem=70.93 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.23it/s]Capturing num tokens (num_tokens=12 avail_mem=70.94 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.23it/s]Capturing num tokens (num_tokens=12 avail_mem=70.94 GB):  97%|█████████▋| 56/58 [00:02<00:00, 37.89it/s]Capturing num tokens (num_tokens=8 avail_mem=70.94 GB):  97%|█████████▋| 56/58 [00:02<00:00, 37.89it/s] Capturing num tokens (num_tokens=4 avail_mem=70.93 GB):  97%|█████████▋| 56/58 [00:02<00:00, 37.89it/s]Capturing num tokens (num_tokens=4 avail_mem=70.93 GB): 100%|██████████| 58/58 [00:02<00:00, 25.91it/s]


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
    Generated text:  Jane. I'm a student of Grade 7. I have a brother. His name is Henry. He is 12 years old. He is also a student of Grade 7. He has a sister. Her name is Jane. She is also a student of Grade 7. We are the only family in our school. We are very happy to have a family. We are very happy to be able to call our parents, grandparents, and other family members when we feel sad. We don't go to a lot of parties. We don't have a lot of toys. We only have two books. But we love
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person, not an animal. Is this statement true or false?
    A. True
    B. False
    C. Cannot be determined
    D. It depends on the context
    E. It depends on the location
    Answer:
    A
    
    Which of the following statements is true?
    A. The definition of a person does not involve gender.
    B. A person is a standard term for a male individual.
    C. A person is a term for a male individual only.
    D. Gender is a characteristic of a person.
    E. Gender is the same as a person.
    
    Answer:
    A
    
    Which of the following statements is true?
    A.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The city is located at the crossroads of four major European rivers: the Seine, the Seine-Darling, the Seine-Loire, and the Loire. The city has been a major European city since the Roman Empire and ancient Gaul, when it was called "Asculus" by the Romans. It was an important center of the industrial revolution and the technological revolution in the 19th century, and the capital of the French Empire.
    
    In the 18th century, the city was called "Basilicum" (which means "basement" in Latin). In 175
    ===============================
    Prompt: The future of AI is
    Generated text:  here and it’s affecting our lives and our environment more than we realize. The future is bright, and we must embrace it. But we must also be mindful of its impact on the environment and its economic implications. We need to be mindful of how we use AI and its potential to both benefit and harm the planet. Here are 5 ways we can use AI to both benefit and harm the environment and economy:
    1. Advancing AI for environmental protection: AI can be used to monitor and detect pollution sources, predict weather patterns, and optimize energy use. This can help in creating sustainable and efficient systems that reduce environmental damage. The future


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and art galleries. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French language. It is also a popular tourist destination, attracting millions of visitors each year. The city is home to many famous French artists, writers, and musicians, and is known for its vibrant nightlife and food scene. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do. This could lead to more efficient and effective decision-making, as well as more personalized and customized experiences for users.
    
    2. Greater reliance on data: AI will continue to rely more heavily on data to learn and improve, and this will likely lead to more data being collected and analyzed
    


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
    Generated text:  [insert first name and last name], and I'm a professional book reviewer. I have a passion for literature, particularly in genres like romance, fantasy, and young adult. I love to share my thoughts on new releases and discuss the underlying themes and motifs that have shaped them over the years. I enjoy being able to engage with people on a personal level and provide them with thoughtful and insightful reviews. My writing style is accessible and engaging, and I'm always eager to learn new things about the world of literature. I'm a hardworking and dedicated person who is always looking for new opportunities to contribute to the literary world. Thank you for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, commonly known as "La ville noire" (the black city) due to the dark colors of the buildings and homes in the city. It is the largest city in France and one of the most populous cities in the world, with a population of over 20 million people. Paris is known for its historical and cultural landmarks, including the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and Champs-Élysées. The city is also famous for its fashion industry and annual cultural events. Paris is a hub for global fashion, art, and entertainment, and it continues to be a major cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, and there are many potential trends and technologies that could shape the development of this field in the coming years. Here are some potential trends and technologies that have the potential to impact AI in significant ways:
    
    1. Increased reliance on robotics and automation: With the rise of automation in manufacturing and industry, there is a lot of potential to see increased reliance on AI and robotics in areas like healthcare, finance, and transportation. This could lead to more efficient processes and greater productivity, but it could also create new challenges around safety and ethical issues.
    
    2. Enhanced AI for healthcare: There is a lot of potential for AI to be used in healthcare


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

     [

    Age

    ],

     [

    Gender

    ]

     [

    Occup

    ation

    ].

     I

     grew

     up

     in

     [

    Your

     hometown

     or

     country

    ],

     and

     now

     I

     am

     a

     [

    Your

     current

     occupation

    ],

     working

     hard

     to

     [

    Your

     current

     goal

    ]

     in

     [

    Your

     industry

    ].

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

    ,

     and

     I

     love

     [

    Your

     hobby

    ,

     sport

    ,

     or

     interest

    ].

     I

     am

     always

     eager

     to

     learn

     and

     grow

    ,

     and

     I

     am

     always

     willing

     to

     listen

     to

     others

    '

     opinions

     and

     ideas

    .

     I

     am

     also

     a

     [

    Your

     favorite

     hobby

     or

     activity

    ],

     and

     I

     enjoy

     spending

     time

     with

     my

     family

     and

     friends

    .

     I

     am

     a

     [

    Your

     relationship

     with

     your

     family

     or

     friends

    ],

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     city

     located

     in

     the

     south

     of

     the

     country

     and

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

     significant

     cultural

     and

     historical

     center

     that

     houses

     many

     of

     the

     country

    's

     major

     museums

    ,

     libraries

    ,

     and

     art

     galleries

    .

     Paris

     is

     known

     for

     its

     op

    ulent

     bou

    lev

    ards

    ,

     beautiful

     gardens

    ,

     and

     rich

     historical

     heritage

    .

     As

     a

     city

     of

     diverse

     cultures

     and

     influences

    ,

     it

     is

     home

     to

     a

     variety

     of

     food

    ,

     art

    ,

     music

    ,

     and

     fashion

     scenes

    .

     The

     city

     is

     also

     home

     to

     several

     international

     organizations

     and

     universities

    ,

     making

     it

     a

     global

     hub

     for

     innovation

     and

     learning

    .

     Based

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     highly

     complex

    ,

     with

     many

     possibilities

     and

     potential

     breakthrough

    s

     that

     could

     shape

     the

     field

     in

     significant

     ways

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     more

     and

     more

     AI

     systems

     are

     deployed

     in

     various

     sectors

    ,

     there

     will

     be

     increasing

     concerns

     about

     their

     impact

     on

     society

    .

     This

     will

     likely

     lead

     to

     greater

     focus

     on

     ethical

     considerations

    ,

     including

     issues

     like

     bias

    ,

     transparency

    ,

     accountability

    ,

     and

     privacy

    .
    


    2

    .

     Enhanced

     human

    -machine

     collaboration

    :

     AI

     is

     likely

     to

     play

     an

     increasingly

     important

     role

     in

     human

    -machine

     collaboration

    ,

     enabling

     machines

     to

     perform

     tasks

     that

     would

     be

     difficult

     or

     impossible

     for

     humans

     to

     do

    .

     This

     could

     lead

     to

     more

     efficient

    



```python
llm.shutdown()
```
