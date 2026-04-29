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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.72it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.71it/s]


    2026-04-29 11:04:06,044 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 11:04:06] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]

    Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:23,  2.19it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:23,  2.19it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:23,  2.19it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:23,  2.19it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:05<00:23,  2.19it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:11,  4.17it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:11,  4.17it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:11,  4.17it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:11,  4.17it/s]

    Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:11,  4.17it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:11,  4.17it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:05<00:05,  7.36it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 12.13it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 12.13it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 12.13it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 12.13it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 12.13it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 12.13it/s]

    Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:02, 12.13it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:02, 12.13it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 18.66it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 18.66it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 18.66it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 18.66it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 18.66it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 18.66it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 18.66it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:01, 18.66it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:01, 18.66it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 26.77it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 26.77it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 26.77it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 26.77it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 26.77it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 26.77it/s]

    Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 26.77it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 26.77it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 26.77it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 34.83it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 34.83it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 34.83it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 34.83it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 34.83it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 34.83it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 34.83it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 34.83it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 41.31it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 41.31it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 41.31it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 41.31it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 41.31it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 41.31it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 41.31it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.21 GB):   2%|▏         | 1/58 [00:00<00:05,  9.83it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.18 GB):   2%|▏         | 1/58 [00:00<00:05,  9.83it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.17 GB):   2%|▏         | 1/58 [00:00<00:05,  9.83it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=116.17 GB):   5%|▌         | 3/58 [00:00<00:04, 11.01it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.17 GB):   5%|▌         | 3/58 [00:00<00:04, 11.01it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.17 GB):   5%|▌         | 3/58 [00:00<00:04, 11.01it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.17 GB):   9%|▊         | 5/58 [00:00<00:04, 12.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.17 GB):   9%|▊         | 5/58 [00:00<00:04, 12.14it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=116.16 GB):   9%|▊         | 5/58 [00:00<00:04, 12.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.16 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.15 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.15 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.15 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.15 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.23it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=116.14 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.14 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.14 GB):  21%|██        | 12/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.14 GB):  21%|██        | 12/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.14 GB):  21%|██        | 12/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.13 GB):  21%|██        | 12/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.13 GB):  21%|██        | 12/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.62it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.12 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.62it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=116.12 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.12 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.12 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.10 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=960 avail_mem=116.11 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.50it/s] Capturing num tokens (num_tokens=896 avail_mem=116.11 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.50it/s]Capturing num tokens (num_tokens=832 avail_mem=116.09 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.50it/s]Capturing num tokens (num_tokens=768 avail_mem=115.94 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.50it/s]Capturing num tokens (num_tokens=768 avail_mem=115.94 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.98it/s]Capturing num tokens (num_tokens=704 avail_mem=115.94 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.98it/s]Capturing num tokens (num_tokens=640 avail_mem=115.93 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.98it/s]

    Capturing num tokens (num_tokens=576 avail_mem=115.93 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.98it/s]Capturing num tokens (num_tokens=512 avail_mem=115.92 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.98it/s]Capturing num tokens (num_tokens=512 avail_mem=115.92 GB):  50%|█████     | 29/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=480 avail_mem=115.79 GB):  50%|█████     | 29/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=448 avail_mem=115.57 GB):  50%|█████     | 29/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=416 avail_mem=115.57 GB):  50%|█████     | 29/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=384 avail_mem=115.57 GB):  50%|█████     | 29/58 [00:01<00:00, 33.24it/s]

    Capturing num tokens (num_tokens=384 avail_mem=115.57 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.22it/s]Capturing num tokens (num_tokens=352 avail_mem=115.38 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.22it/s]Capturing num tokens (num_tokens=320 avail_mem=115.37 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.22it/s]Capturing num tokens (num_tokens=288 avail_mem=115.37 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.22it/s]Capturing num tokens (num_tokens=256 avail_mem=115.37 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.22it/s]Capturing num tokens (num_tokens=256 avail_mem=115.37 GB):  64%|██████▍   | 37/58 [00:01<00:00, 30.60it/s]Capturing num tokens (num_tokens=240 avail_mem=115.37 GB):  64%|██████▍   | 37/58 [00:01<00:00, 30.60it/s]Capturing num tokens (num_tokens=224 avail_mem=115.36 GB):  64%|██████▍   | 37/58 [00:01<00:00, 30.60it/s]

    Capturing num tokens (num_tokens=208 avail_mem=115.36 GB):  64%|██████▍   | 37/58 [00:01<00:00, 30.60it/s]Capturing num tokens (num_tokens=192 avail_mem=115.36 GB):  64%|██████▍   | 37/58 [00:01<00:00, 30.60it/s]Capturing num tokens (num_tokens=192 avail_mem=115.36 GB):  71%|███████   | 41/58 [00:01<00:00, 23.48it/s]Capturing num tokens (num_tokens=176 avail_mem=117.96 GB):  71%|███████   | 41/58 [00:01<00:00, 23.48it/s]

    Capturing num tokens (num_tokens=160 avail_mem=117.96 GB):  71%|███████   | 41/58 [00:01<00:00, 23.48it/s]Capturing num tokens (num_tokens=144 avail_mem=117.95 GB):  71%|███████   | 41/58 [00:01<00:00, 23.48it/s]Capturing num tokens (num_tokens=128 avail_mem=117.95 GB):  71%|███████   | 41/58 [00:01<00:00, 23.48it/s]Capturing num tokens (num_tokens=128 avail_mem=117.95 GB):  78%|███████▊  | 45/58 [00:01<00:00, 25.65it/s]Capturing num tokens (num_tokens=112 avail_mem=117.95 GB):  78%|███████▊  | 45/58 [00:01<00:00, 25.65it/s]Capturing num tokens (num_tokens=96 avail_mem=117.94 GB):  78%|███████▊  | 45/58 [00:01<00:00, 25.65it/s] Capturing num tokens (num_tokens=80 avail_mem=117.94 GB):  78%|███████▊  | 45/58 [00:01<00:00, 25.65it/s]Capturing num tokens (num_tokens=64 avail_mem=117.94 GB):  78%|███████▊  | 45/58 [00:01<00:00, 25.65it/s]Capturing num tokens (num_tokens=64 avail_mem=117.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=48 avail_mem=117.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 28.67it/s]

    Capturing num tokens (num_tokens=32 avail_mem=117.93 GB):  84%|████████▍ | 49/58 [00:02<00:00, 28.67it/s]Capturing num tokens (num_tokens=28 avail_mem=117.93 GB):  84%|████████▍ | 49/58 [00:02<00:00, 28.67it/s]Capturing num tokens (num_tokens=24 avail_mem=117.92 GB):  84%|████████▍ | 49/58 [00:02<00:00, 28.67it/s]Capturing num tokens (num_tokens=24 avail_mem=117.92 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.28it/s]Capturing num tokens (num_tokens=20 avail_mem=117.92 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.28it/s]Capturing num tokens (num_tokens=16 avail_mem=117.92 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.28it/s]Capturing num tokens (num_tokens=12 avail_mem=117.91 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.28it/s]

    Capturing num tokens (num_tokens=8 avail_mem=118.80 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.28it/s] Capturing num tokens (num_tokens=8 avail_mem=118.80 GB):  98%|█████████▊| 57/58 [00:02<00:00, 24.47it/s]Capturing num tokens (num_tokens=4 avail_mem=118.80 GB):  98%|█████████▊| 57/58 [00:02<00:00, 24.47it/s]Capturing num tokens (num_tokens=4 avail_mem=118.80 GB): 100%|██████████| 58/58 [00:02<00:00, 24.61it/s]


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
    Generated text:  Jonas and I am a 12 year old boy. I have been told that I have ADHD. Is there anything that can be done to help the ADHD? I do not want to take a pill, I just want to take care of myself. What can I do to help?
    
    As a 12-year-old boy, it's important to understand that ADHD (Attention Deficit Hyperactivity Disorder) is a condition that affects the body's ability to concentrate and stay focused. While there are ways to manage ADHD symptoms and improve overall mental health, it's not recommended to take medication as a primary solution.
    
    If you are experiencing symptoms
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. Let x be the number of bases. Each base costs $100 million,000,000 to build. However, each base also requires an annual maintenance cost of $20 million. The president wants to ensure that the total maintenance cost over a year is less than the total cost of the bases. 
    
    What is the maximum number of bases the president can afford to have if the total maintenance cost over a year must be less than the total cost of the bases?
    
    To determine the maximum number of bases the president can afford, we need to analyze the total maintenance
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The weather is very hot in Paris in summer. It is hot all day. It is very hot in summer in Paris. When the sun is strong, the air is hot and the weather is hot and sunny. The air is dry. There are not many trees in Paris. There are not many trees in Paris. The air is dirty. There are not many flowers in Paris. There are not many flowers in Paris. There are not many birds. There are not many birds. The sky is green and clear. The sky is green and clear. The sky is blue and blue. The sky is blue and blue. The
    ===============================
    Prompt: The future of AI is
    Generated text:  up in the air, and many companies are eager to get into it. For a while, I was skeptical about the prospects for AI and wonder if it will really bring about a shift in the way we work. But with all the promises and challenges, it seems that there is no time to waste. Here is a summary of the ways that AI is transforming the world.
    One of the most significant ways that AI is transforming the world is through its ability to automate routine tasks and reduce human errors. With the right software and algorithms, machines can perform tasks that were previously carried out by humans, such as data analysis, customer service, and


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your character here]. And what can you tell me about your company? I'm excited to learn more about you and your work. What can you tell me about your company? I'm excited to learn more about you and your work. What can you tell me about your company? I'm excited to learn more about you and your work. What can you tell me about your company? I'm excited to learn more
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and economic hub, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. Paris is a popular tourist destination and a major center for art, music, and literature. It is home to many famous museums, including the Musée d'Orsay, the Louvre, and the Musée Rodin. The city is also known for its cuisine, with a variety of traditional dishes and international flavors. Paris is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased integration with other technologies: AI will continue to be integrated with other technologies such as IoT, blockchain, and quantum computing, which will create new opportunities for AI to be used in new and innovative ways.
    
    2. Enhanced privacy and security: As AI becomes more integrated with other technologies, there will be a need to address privacy and security concerns. This will require the development of new algorithms and techniques to ensure that AI systems are not vulnerable to cyber attacks or data breaches.
    
    3
    


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
    Generated text:  [Name] and I am a [job title or profession]. I have been [number of years working in this field], and I have a passion for [mention a specific activity or hobby]. I love [mention something about my interests or hobbies], and I am always looking for ways to [mention something new or exciting about my career]. I am a [occupation] and I believe that [mention an important value or belief]. I am always [mention something positive or positive qualities to have]. I am [mention a positive trait or personal characteristic]. I am [mention an attribute or characteristic that sets me apart from others]. And my [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, an iconic and historic city located in the south of the country.
    The city is known for its rich history, vibrant culture, and beautiful architecture. It's home to the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, among other landmarks. The city also has a long history dating back to the Roman Empire, and remains a cultural hub for France and Europe. With its world-class cuisine, fashion, and art scene, Paris is a popular destination for tourists and locals alike. The city's climate is temperate, with mild winters and hot summers, making it a pleasant place to live and visit.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by a variety of factors, including advances in computing power, data availability, and the ability to train AI to learn and adapt to new situations. Here are some possible future trends in AI:
    
    1. Increased integration with other industries: As AI becomes more ubiquitous, it will be increasingly integrated into various industries, including healthcare, finance, manufacturing, and transportation. This could lead to more efficient and effective use of resources, as well as new opportunities for collaboration and innovation.
    
    2. More personalized and context-aware AI: AI will become more capable of understanding and adapting to the unique needs and behaviors of individuals, leading to more personalized


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

    Age

    ]

     year

     old

     [

    Gender

    ]

     [

    Appearance

    ]

     person

    .

     I

     have

     a

     [

    Occup

    ation

    ],

     but

     I

     find

     it

     challenging

     to

     articulate

     my

     full

     potential

     due

     to

     [

    Challenge

    ].

     I

     am

     currently

     [

    Position

    ]

     in

     this

     industry

    ,

     but

     I

     am

     always

     looking

     for

     new

     opportunities

     to

     grow

     and

     improve

     myself

    .

     I

     am

     always

     seeking

     advice

     and

     feedback

     on

     how

     to

     improve

     my

     skills

     and

     abilities

    .

     My

     goal

     is

     to

     achieve

     [

    Objective

    ],

     and

     I

     am

     dedicated

     to

     [

    Goal

    ].

     I

     am

     passionate

     about

     [

    Interest

    ],

     and

     I

     am

     always

     eager

     to

     learn

     more

     about

     it

    .

     My

     love

     for

     [

    Interest

    ]

     has

     taken

     me

     on

     a

     journey

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

      
    


    Provide

     a

     concise

     factual

     statement

     about

     France

    ’s

     capital

     city

    .

     The

     capital

     of

     France

     is

     Paris

    .

      
    


    Select

     from

     the

     following

    .


    a

    ).

     no

    .


    b

    ).

     yes

    .


    Is

     the

     question

     re

    ph

    r

    ased

    ?

     a

    ).

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     possibilities

     for

     innovation

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     Use

     of

     AI

     in

     Healthcare

    :

     The

     healthcare

     industry

     is

     one

     of

     the

     sectors

     that

     is

     most

     likely

     to

     benefit

     from

     AI

    .

     AI

     has

     the

     potential

     to

     improve

     diagnosis

    ,

     treatment

    ,

     and

     patient

     outcomes

    .

     AI

    -powered

     tools

     like

     chat

    bots

    ,

     virtual

     assistants

    ,

     and

     robots

     in

     the

     healthcare

     industry

     could

     make

     patients

     more

     comfortable

    ,

     reduce

     waiting

     times

    ,

     and

     save

     healthcare

     providers

     time

     and

     money

    .
    


    2

    .

     Increased

     Use

     of

     AI

     in

     Financial

     Services

    :

     The

     financial

     services

     industry

     is

     another

     sector

     that

     is

     likely

     to

     benefit

     from

     AI

    .

     AI

     can

     be

     used

     to

     detect

     fraudulent

     activities

    



```python
llm.shutdown()
```
