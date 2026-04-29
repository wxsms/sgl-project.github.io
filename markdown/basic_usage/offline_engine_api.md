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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.47it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.46it/s]


    2026-04-29 08:35:43,807 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 08:35:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:42,  4.95s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:42,  4.95s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:42,  4.95s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:13,  1.33s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:13,  1.33s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:05<01:13,  1.33s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:20,  2.43it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:20,  2.43it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:20,  2.43it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:20,  2.43it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:11,  4.26it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:11,  4.26it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:11,  4.26it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:11,  4.26it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:11,  4.26it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:05<00:05,  7.36it/s] Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:05<00:05,  7.36it/s]

    Compiling num tokens (num_tokens=832):  24%|██▍       | 14/58 [00:05<00:05,  7.36it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 17.48it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 17.48it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 17.48it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 17.48it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 17.48it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 17.48it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:01, 17.48it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:05<00:01, 17.48it/s]Compiling num tokens (num_tokens=416):  41%|████▏     | 24/58 [00:05<00:01, 17.48it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 25.87it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 25.87it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 25.87it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 25.87it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 25.87it/s]

    Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 25.87it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 25.87it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 34.50it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 34.50it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 34.50it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 34.50it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:06<00:00, 34.50it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:06<00:00, 34.50it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:06<00:00, 34.50it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:06<00:00, 39.12it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:06<00:00, 39.12it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:06<00:00, 39.12it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:06<00:00, 39.12it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:06<00:00, 39.12it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:06<00:00, 39.12it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:06<00:00, 39.12it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:06<00:00, 39.12it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:06<00:00, 39.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00, 47.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.00 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.00 GB):   3%|▎         | 2/58 [00:00<00:03, 15.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.00 GB):   3%|▎         | 2/58 [00:00<00:03, 15.30it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.63 GB):   3%|▎         | 2/58 [00:00<00:03, 15.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.63 GB):   7%|▋         | 4/58 [00:00<00:04, 11.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.63 GB):   7%|▋         | 4/58 [00:00<00:04, 11.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.45 GB):   7%|▋         | 4/58 [00:00<00:04, 11.90it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=57.45 GB):  10%|█         | 6/58 [00:00<00:03, 13.08it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.44 GB):  10%|█         | 6/58 [00:00<00:03, 13.08it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.43 GB):  10%|█         | 6/58 [00:00<00:03, 13.08it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=57.43 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.43 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.64it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.43 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.64it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.43 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.42 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.14it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=57.42 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.42 GB):  21%|██        | 12/58 [00:00<00:03, 13.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.42 GB):  21%|██        | 12/58 [00:00<00:03, 13.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.42 GB):  21%|██        | 12/58 [00:00<00:03, 13.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.42 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.41 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.41 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.24it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=57.41 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.29 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:01<00:02, 14.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:01<00:02, 14.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:01<00:02, 14.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:01<00:02, 14.14it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:01<00:02, 14.14it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  31%|███       | 18/58 [00:01<00:02, 14.14it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.30it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.30it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.30it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.30it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.30it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 22.30it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.74it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.74it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.74it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.74it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.74it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.74it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.58it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.58it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.61it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.61it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.61it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.61it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.61it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.61it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.23it/s]

    Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:02<00:00, 42.23it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:02<00:00, 42.23it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:02<00:00, 42.23it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  91%|█████████▏| 53/58 [00:02<00:00, 40.31it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:02<00:00, 40.31it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:02<00:00, 40.31it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:02<00:00, 40.31it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:02<00:00, 40.31it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:02<00:00, 40.31it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:02<00:00, 41.55it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:02<00:00, 26.21it/s]


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
    Generated text:  Ben. I’m a bit of a nerd, and I have a passion for coding. I've been coding for 6 years now, and I'm still learning every day. I love being able to create something that is useful and beautiful.
    I started coding back in high school, using my TI-83 Plus and TI-84 Plus calculators. I also worked as a part-time software developer for a couple of years. In college, I worked as a data analyst for a local company, where I was responsible for creating reports and graphs using various software tools.
    My favorite language is Python, and I have a deep knowledge
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. The average age of five people that were in his first term is 70 years. The average age of six people that were in his second term is 62 years. What is the average age of all the people that were in his second term?
    Answer Choices: (A) 60 (B) 61 (C) 62 (D) 63 (E) 64
    
    To find the average age of all the people in the second term, we need to consider the total age of the people in both terms and then divide by the total number of people
    ===============================
    Prompt: The capital of France is
    Generated text:  __________.
    A. Paris
    B. London
    C. Rome
    D. New York
    Answer:
    
    A
    
    The capital of France is __________.
    A. Paris
    B. London
    C. Rome
    D. New York
    Answer:
    
    A
    
    In the government budget, which category does the fiscal deficit specifically fall under? 
    A. Operating budget
    B. Capital budget
    C. Capital surplus
    D. Operating surplus
    Answer:
    
    C
    
    Which of the following statements about the actuarial table is incorrect?
    A. The actuarial table is an important tool for life insurance companies to calculate and verify the value
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but as an AI researcher with decades of experience, what’s your key learning takeaway that has allowed you to help guide the growth of the industry?
    
    As an AI researcher, I believe that the key learning takeaway is understanding the multifaceted nature of AI and its impact on society. AI has the potential to revolutionize many industries and solve complex problems, but it also poses significant risks, ethical concerns, and unintended consequences. As an AI researcher, I have been committed to working to bridge these gaps and ensure that AI is used ethically and responsibly to improve the quality of life for all people.
    
    Additionally, I have a deep understanding


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has always been [Positive Trait]. I'm passionate about [What I Love to Do]. I'm a [Favorite Hobby] that I enjoy [How I Enjoy It]. I'm a [Favorite Book] that I read every day. I'm a [Favorite Movie] that I watch every weekend. I'm a [Favorite Music Artist] that I listen to every night. I'm a [Favorite Sport] that I play [How I Play It]. I'm a [Favorite Food] that I love [How I Love
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife, fashion industry, and world-renowned cuisine. The city is a major hub for business, education, and entertainment, and is a popular tourist destination. It is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. Paris is a city of art, culture, and innovation, and is a must-visit destination for anyone interested in the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations in a more natural way. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely
    


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
    Generated text:  Emily and I'm a 35-year-old software engineer with a passion for technology and creativity. I'm always on the lookout for new ideas and fresh ways to challenge myself. And I love to collaborate with like-minded individuals to achieve our goals. I believe that creativity and hard work lead to success, and I'm excited to bring my skills and experience to your team. Thank you! Emily. That sounds like a great character profile. Can you share some of your favorite programming languages or tools that you use? Certainly! I'm a huge proponent of using languages like Python and JavaScript for my work. These languages are both powerful and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the historic center of the country. 
    
    A. Incorrect B. Correct
    
    B. Correct
    
    The capital of France is indeed Paris, the historic center of the country. It is a vibrant and culturally rich city known for its art, history, and cuisine. Paris, with its towers, bridges, and museums, is a popular tourist destination, and it is home to numerous world-renowned landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. Paris also has a strong military history, with the city being the birthplace of the French military and numerous other
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but some potential trends that could be expected to shape the technology's trajectory include:
    
    1. Increased integration with human decision-making: AI is likely to become more integrated with human decision-making processes, with more AI systems capable of understanding and interpreting human emotions and biases. This could lead to more nuanced and context-aware AI systems that can make better decisions.
    
    2. Greater emphasis on ethical AI: As more AI systems become autonomous, there is a growing need for them to be designed with ethical considerations in mind. This could include rules and regulations around privacy, fairness, and accountability.
    
    3. Better data privacy: As AI systems become more powerful


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

    'm

     a

    /an

     [

    职业

    ]

     with

     [

    number

    ]

     years

     of

     experience

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?
    


    Your

     response

     should

     be

     

    1

    0

    0

     words

     or

     less

    .

     As

     a

     professional

     consultant

     with

     over

     

    8

     years

     of

     experience

     in

     marketing

     and

     public

     relations

    ,

     I

     have

     a

     proven

     track

     record

     of

     success

     in

     helping

     businesses

     of

     all

     sizes

     and

     industries

     grow

     and

     achieve

     their

     marketing

     and

     public

     relations

     goals

    .

     My

     expertise

     lies

     in

     crafting

     compelling

     strategies

     that

     resonate

     with

     our

     clients

     and

     help

     them

     succeed

     in

     their

     markets

    .

     I

     am

     skilled

     in

     using

     various

     tools

     and

     techniques

     to

     achieve

     our

     marketing

     and

     public

     relations

     objectives

    ,

     and

     I

     am

     committed

     to

     using

     my

     skills

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

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

     It

     is

     also

     famous

     for

     its

     sophisticated

     culture

     and

     annual

     cultural

     festivals

     such

     as

     the

     Tour

     de

     France

     bicycle

     race

     and

     the

     Hundred

     Days

    '

     Revolution

    .

     Its

     status

     as

     a

     major

     financial

     center

    ,

     tourism

     hub

    ,

     and

     cultural

     center

     has

     made

     it

     one

     of

     the

     most

     important

     cities

     in

     Europe

    .

     Paris

     is

     home

     to

     many

     world

    -ren

    owned

     artists

     and

     designers

    ,

     and

     is

     considered

     one

     of

     the

     world

    's

     most

     beautiful

     cities

    .

     The

     French

     language

     is

     also

     spoken

     throughout

     the

     country

     and

     is

     a

     widely

     recognized

     language

     of

     international

     significance

    .

     Paris

     was

     founded

     in

     the

     

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     driven

     by

     advances

     in

     machine

     learning

     and

     deep

     learning

    ,

     as

     well

     as

     new

     technologies

     such

     as

     quantum

     computing

     and

     bi

    otechnology

    .

     Additionally

    ,

     there

     may

     be

     a

     growing

     emphasis

     on

     ethical

     and

     social

     implications

     of

     AI

    ,

     with

     the

     aim

     of

     ensuring

     that

     AI

     is

     used

     to

     serve

     human

     interests

     and

     promote

     positive

     social

     outcomes

    .

     Finally

    ,

     there

     may

     be

     a

     focus

     on

     creating

     more

     advanced

     and

     flexible

     AI

     systems

     that

     can

     adapt

     to

     changing

     circumstances

     and

     remain

     effective

     in

     handling

     new

     tasks

     and

     challenges

    .

     These

     trends

     suggest

     that

     AI

     will

     continue

     to

     evolve

     and

     grow

    ,

     with

     new

     applications

     emerging

     regularly

    ,

     as

     technologies

     and

     algorithms

     continue

     to

     evolve

     and

     improve

    .

     It

     is

     important

     for

     policymakers

    ,

     researchers

    ,

    



```python
llm.shutdown()
```
