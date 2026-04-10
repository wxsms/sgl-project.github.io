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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.92it/s]


    2026-04-10 19:50:16,046 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 19:50:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:21,  2.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:21,  2.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:21,  2.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:21,  2.47it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:10,  4.67it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:10,  4.67it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:10,  4.67it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:10,  4.67it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:03<00:10,  4.67it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:03<00:10,  4.67it/s]

    Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:03<00:10,  4.67it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:03<00:10,  4.67it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 11.36it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 11.36it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 11.36it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 11.36it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 11.36it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 14.58it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 14.58it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 14.58it/s]

    Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 14.58it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 14.58it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 14.58it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 22.92it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 22.92it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 22.92it/s]

    Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 22.92it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 22.92it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:00, 26.14it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:00, 26.14it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:00, 26.14it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:00, 26.14it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:00, 26.14it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:00, 26.14it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 30.26it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 33.72it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 33.72it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 33.72it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 33.72it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 33.72it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 33.72it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 34.47it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 34.47it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 34.47it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 34.47it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 34.47it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 34.47it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 34.47it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:04<00:00, 38.76it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:04<00:00, 38.76it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 38.76it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:04<00:00, 38.76it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:04<00:00, 38.76it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:04<00:00, 38.76it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.26 GB):   3%|▎         | 2/58 [00:00<00:04, 12.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.25 GB):   3%|▎         | 2/58 [00:00<00:04, 12.00it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.25 GB):   3%|▎         | 2/58 [00:00<00:04, 12.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.25 GB):   7%|▋         | 4/58 [00:00<00:04, 13.22it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.25 GB):   7%|▋         | 4/58 [00:00<00:04, 13.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.24 GB):   7%|▋         | 4/58 [00:00<00:04, 13.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.24 GB):  10%|█         | 6/58 [00:00<00:03, 14.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.24 GB):  10%|█         | 6/58 [00:00<00:03, 14.69it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=57.22 GB):  10%|█         | 6/58 [00:00<00:03, 14.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.22 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.17 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.29it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=57.21 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.21 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.76it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.17 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.76it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.20 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.76it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.19 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.76it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.19 GB):  22%|██▏       | 13/58 [00:00<00:02, 15.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.19 GB):  22%|██▏       | 13/58 [00:00<00:02, 15.90it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=57.18 GB):  22%|██▏       | 13/58 [00:00<00:02, 15.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.17 GB):  22%|██▏       | 13/58 [00:00<00:02, 15.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.17 GB):  28%|██▊       | 16/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.18 GB):  28%|██▊       | 16/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.16 GB):  28%|██▊       | 16/58 [00:01<00:02, 18.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.15 GB):  28%|██▊       | 16/58 [00:01<00:02, 18.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.14 GB):  28%|██▊       | 16/58 [00:01<00:02, 18.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.14 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.12 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.66it/s]

    Capturing num tokens (num_tokens=960 avail_mem=57.13 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.66it/s] Capturing num tokens (num_tokens=896 avail_mem=57.11 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.66it/s]Capturing num tokens (num_tokens=832 avail_mem=57.11 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.66it/s]Capturing num tokens (num_tokens=832 avail_mem=57.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.22it/s]Capturing num tokens (num_tokens=768 avail_mem=57.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.22it/s]Capturing num tokens (num_tokens=704 avail_mem=57.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.22it/s]Capturing num tokens (num_tokens=640 avail_mem=57.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.22it/s]Capturing num tokens (num_tokens=576 avail_mem=57.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.22it/s]

    Capturing num tokens (num_tokens=576 avail_mem=57.11 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.22it/s]Capturing num tokens (num_tokens=512 avail_mem=57.09 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.22it/s]Capturing num tokens (num_tokens=480 avail_mem=57.10 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.22it/s]Capturing num tokens (num_tokens=448 avail_mem=57.10 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.22it/s]Capturing num tokens (num_tokens=416 avail_mem=57.07 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.22it/s]Capturing num tokens (num_tokens=416 avail_mem=57.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=384 avail_mem=57.09 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=352 avail_mem=57.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=320 avail_mem=57.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.30it/s]Capturing num tokens (num_tokens=288 avail_mem=57.06 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.30it/s]

    Capturing num tokens (num_tokens=288 avail_mem=57.06 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.54it/s]Capturing num tokens (num_tokens=256 avail_mem=57.06 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.54it/s]Capturing num tokens (num_tokens=240 avail_mem=57.05 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.54it/s]Capturing num tokens (num_tokens=224 avail_mem=57.04 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.54it/s]Capturing num tokens (num_tokens=208 avail_mem=57.03 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.54it/s]Capturing num tokens (num_tokens=208 avail_mem=57.03 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=192 avail_mem=57.03 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=176 avail_mem=57.02 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=160 avail_mem=57.02 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.28it/s]

    Capturing num tokens (num_tokens=144 avail_mem=57.01 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.28it/s]

    Capturing num tokens (num_tokens=144 avail_mem=57.01 GB):  76%|███████▌  | 44/58 [00:02<00:00, 16.41it/s]Capturing num tokens (num_tokens=128 avail_mem=57.01 GB):  76%|███████▌  | 44/58 [00:02<00:00, 16.41it/s]Capturing num tokens (num_tokens=112 avail_mem=57.01 GB):  76%|███████▌  | 44/58 [00:02<00:00, 16.41it/s]Capturing num tokens (num_tokens=96 avail_mem=57.00 GB):  76%|███████▌  | 44/58 [00:02<00:00, 16.41it/s] Capturing num tokens (num_tokens=96 avail_mem=57.00 GB):  81%|████████  | 47/58 [00:02<00:00, 18.47it/s]Capturing num tokens (num_tokens=80 avail_mem=57.00 GB):  81%|████████  | 47/58 [00:02<00:00, 18.47it/s]Capturing num tokens (num_tokens=64 avail_mem=57.00 GB):  81%|████████  | 47/58 [00:02<00:00, 18.47it/s]

    Capturing num tokens (num_tokens=48 avail_mem=56.99 GB):  81%|████████  | 47/58 [00:02<00:00, 18.47it/s]Capturing num tokens (num_tokens=48 avail_mem=56.99 GB):  86%|████████▌ | 50/58 [00:02<00:00, 16.10it/s]Capturing num tokens (num_tokens=32 avail_mem=56.99 GB):  86%|████████▌ | 50/58 [00:02<00:00, 16.10it/s]Capturing num tokens (num_tokens=28 avail_mem=56.99 GB):  86%|████████▌ | 50/58 [00:02<00:00, 16.10it/s]

    Capturing num tokens (num_tokens=24 avail_mem=56.98 GB):  86%|████████▌ | 50/58 [00:02<00:00, 16.10it/s]Capturing num tokens (num_tokens=24 avail_mem=56.98 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.52it/s]Capturing num tokens (num_tokens=20 avail_mem=56.98 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.52it/s]Capturing num tokens (num_tokens=16 avail_mem=56.98 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.52it/s]Capturing num tokens (num_tokens=12 avail_mem=56.97 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.52it/s]Capturing num tokens (num_tokens=8 avail_mem=56.97 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.52it/s] Capturing num tokens (num_tokens=4 avail_mem=56.97 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.52it/s]Capturing num tokens (num_tokens=4 avail_mem=56.97 GB): 100%|██████████| 58/58 [00:02<00:00, 22.03it/s]Capturing num tokens (num_tokens=4 avail_mem=56.97 GB): 100%|██████████| 58/58 [00:02<00:00, 20.39it/s]


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
    Generated text:  Mike and I'm a passionate, confident, and talented professional who has over 10 years of experience in accounting, tax preparation, and financial management.
    What's something I can do that really improves my professional skills and can help me better serve you? There's no one-size-fits-all answer to this question, as the specific skills and experiences that are most valuable can vary depending on your individual needs and goals.
    However, I can suggest some potential areas that could help you improve your professional skills:
    
    1. Attend industry conferences and events: By attending relevant conferences and events, you can network with professionals in your field and gain access to
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 37 years younger than Bob. If Bob is currently 50 years old, how old will the president be when he retires and receives $1,000,000 at the age of 80? To determine how old the president will be when he retires, we need to first calculate his current age. We know that the president is currently 37 years younger than Bob. Given that Bob is currently 50 years old, we can find his current age by subtracting 37 from 50.
    
    \[
    \text{President's current age} = 50
    ===============================
    Prompt: The capital of France is
    Generated text:  ____. 
    A. London 
    B. Paris 
    C. Rome 
    D. Moscow 
    Answer:
    B
    
    The report of the 18th National Congress of the Communist Party of China pointed out that the market should play a decisive role in the allocation of resources, better发挥the role of the government, and establish a ___ economic system.
    A. Market-oriented, rule-of-law-based, modernized
    B. Market-oriented, regulatory-based, modernized
    C. Market-oriented, rule-of-law-based, open
    D. Market-oriented, regulatory-based, open
    Answer:
    A
    
    On August 10, 2
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about the emergence of intelligent machines, but also about the impact of such machines on society. What are the potential consequences of AI development and its interactions with human society? AI has the potential to transform every aspect of society, from healthcare to finance to transportation. However, there are also potential risks and ethical concerns that need to be addressed. 
    
    In terms of healthcare, AI has the potential to improve diagnostics and treatment outcomes, while also enabling the development of personalized medicine. However, there are concerns about the potential for AI to perpetuate bias and discrimination in the healthcare system. In finance, AI has the potential to improve fraud detection and


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your job or profession]. I enjoy [insert a brief description of your hobbies or interests]. I'm [insert a brief description of your personality or character]. Thank you for taking the time to meet me. What's your name? What's your job title? What's your company name? What's your job? What's your hobbies or interests? What's your personality or character? I'm [insert a brief
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its cuisine, fashion, and art. Paris is a cultural and economic hub of France and a major tourist destination. It is home to many world-renowned museums, theaters, and art galleries. The city is also known for its annual festivals and celebrations, such as the Eiffel Tower Festival and the Lou
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be an increased focus on ethical considerations. This will include issues such as bias, transparency, and accountability. AI developers will need to be more mindful of the potential impact of their technology on society and work to ensure that it is used in a way that is fair and beneficial.
    
    2. Greater integration with other technologies: AI is already being
    


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
    Generated text:  [Name], I'm a [occupation], and I'm really excited to meet you. What can you tell me about yourself? I enjoy [interest, hobby, or interest in life] and I'm always looking for new experiences. I'm always ready to learn, grow, and improve, and I'm always willing to try new things. What excites you the most about the future? I'm really excited to see what new adventures and experiences I'll have in the future. What are your goals for the future? I'm really focused on achieving my goals and making the most of my life. How do you see the world?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city of lights, and a bustling metropolis known for its rich history, art, and cuisine. Paris is known for its iconic landmarks like the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, as well as its vibrant cultural scene and world-renowned museums like the Musée d'Orsay and the Musée national d'art moderne. It is also home to numerous museums, theaters, and restaurants, making it a popular tourist destination. Paris is a city that is steeped in tradition and has become a global cultural hub, attracting people from all over the world with its beautiful architecture, charming
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, with many possibilities and possibilities for growth and innovation. Here are some possible trends in AI:
    
    1. Increased availability of AI-powered tools: As AI technology continues to improve, we expect to see more sophisticated and versatile AI tools available on the market. These tools will be designed to perform a wide range of tasks, from image and speech recognition to natural language processing to predictive analytics.
    
    2. AI for healthcare: With the increasing demand for AI-powered healthcare solutions, we can expect to see an increase in the use of AI in medical imaging, drug discovery, and other healthcare applications. AI-powered algorithms will be used to analyze medical images


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

     an

     [

    occupation

     or

     profession

    ]

     from

     [

    City

    ].

     I

    'm

     a

     [

    job

     title

    ]

     [

    initial

    s

    ].

     I

    've

     been

     in

     the

     business

     for

     [

    number

    ]

     years

     now

     and

     have

     [

    number

    ]

     years

     of

     experience

    .

     I

     started

     my

     career

     in

     [

    past

     job

    ],

     but

     I

     have

     always

     been

     dedicated

     to

     [

    reason

     why

     I

     love

     this

     job

    ].

     I

     am

     [

    number

    ]

     years

     into

     my

     career

    ,

     and

     I

    'm

     always

     learning

     and

     growing

    .

     I

    'm

     excited

     to

     share

     my

     experiences

     and

     experience

     with

     others

    .

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

     an

     [

    occupation

     or

     profession

    ]

     with

     [

    number

    ]

     years

     of

     experience

     in

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     bustling

     met

    ropolis

     with

     a

     rich

     history

     and

     vibrant

     culture

    .
    


    Please

     provide

     a

     brief

     description

     of

     Paris

    's

     architecture

     and

     landmarks

    ,

     including

     examples

     of

     notable

     buildings

     and

     iconic

     landmarks

    .

     Also

    ,

     please

     describe

     the

     cuisine

     and

     attractions

     of

     Paris

    .

     Be

     sure

     to

     include

     information

     about

     the

     city

    's

     climate

    ,

     weather

     patterns

    ,

     and

     notable

     outdoor

     sites

    .
    


    Finally

    ,

     please

     provide

     a

     general

     overview

     of

     Paris

    's

     culture

     and

     history

    ,

     including

     its

     significance

     as

     a

     destination

     for

     tourists

     and

     its

     contributions

     to

     French

     and

     global

     culture

    .

     Your

     response

     should

     be

     detailed

     and

     comprehensive

    ,

     and

     should

     include

     specific

     information

     about

     Paris

    's

     current

     population

    ,

     population

     density

    ,

     and

     cultural

     demographics

    .

     Additionally

    ,

     please

     provide

     an

     analysis

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     unpredictable

    ,

     and

     it

     will

     likely

     continue

     to

     evolve

     and

     change

     at

     an

     alarming

     rate

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     collaboration

     between

     humans

     and

     machines

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     collaboration

     between

     humans

     and

     machines

     in

     a

     variety

     of

     industries

    .

     This

     could

     lead

     to

     more

     personalized

     and

     efficient

     solutions

    ,

     as

     well

     as

     more

     effective

     communication

     between

     the

     two

    .
    


    2

    .

     Greater

     emphasis

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     prevalent

     in

     our

     lives

    ,

     there

     is

     a

     growing

     emphasis

     on

     ethical

     considerations

    .

     This

     could

     lead

     to

     the

     development

     of

     new

     technologies

     that

     prioritize

     the

     well

    -being

     of

     humans

     and

     the

     environment

    ,

     as

     well

     as

    



```python
llm.shutdown()
```
