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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 11:40:56] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.12it/s]


    2026-04-24 11:41:01,155 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 11:41:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:23,  2.25it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.25it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.08it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.08it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.08it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.08it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.08it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.08it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.08it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.08it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.20it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.20it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.20it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.20it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.20it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.20it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.20it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.20it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]

    Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:03<00:01, 24.82it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 31.87it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 31.87it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 31.87it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 31.87it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 31.87it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 31.87it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 31.87it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 31.87it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 38.79it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 38.79it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 38.79it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 38.79it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 38.79it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 38.79it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 38.79it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 38.79it/s]

    Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 38.79it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 46.64it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 46.64it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 46.64it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 46.64it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.47 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.42 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.41 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.41 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.41 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.41 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.74it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.74it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.74it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=116.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.34 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.34 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.34 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.34 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=116.32 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=960 avail_mem=116.33 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s] Capturing num tokens (num_tokens=896 avail_mem=116.33 GB):  31%|███       | 18/58 [00:00<00:01, 34.97it/s]Capturing num tokens (num_tokens=896 avail_mem=116.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=832 avail_mem=116.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=768 avail_mem=116.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=704 avail_mem=116.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=640 avail_mem=116.31 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=576 avail_mem=116.31 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=576 avail_mem=116.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.49it/s]Capturing num tokens (num_tokens=512 avail_mem=116.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.49it/s]

    Capturing num tokens (num_tokens=480 avail_mem=116.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.49it/s]Capturing num tokens (num_tokens=448 avail_mem=116.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.49it/s]Capturing num tokens (num_tokens=416 avail_mem=116.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.49it/s]Capturing num tokens (num_tokens=384 avail_mem=116.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.49it/s]Capturing num tokens (num_tokens=384 avail_mem=116.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.81it/s]Capturing num tokens (num_tokens=352 avail_mem=116.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.81it/s]Capturing num tokens (num_tokens=320 avail_mem=116.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.81it/s]Capturing num tokens (num_tokens=288 avail_mem=116.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.81it/s]Capturing num tokens (num_tokens=256 avail_mem=116.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.81it/s]Capturing num tokens (num_tokens=240 avail_mem=116.29 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.81it/s]

    Capturing num tokens (num_tokens=240 avail_mem=116.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.64it/s]Capturing num tokens (num_tokens=224 avail_mem=116.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.64it/s]Capturing num tokens (num_tokens=208 avail_mem=116.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.64it/s]Capturing num tokens (num_tokens=192 avail_mem=116.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.64it/s]Capturing num tokens (num_tokens=176 avail_mem=116.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.64it/s]Capturing num tokens (num_tokens=160 avail_mem=116.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.64it/s]Capturing num tokens (num_tokens=160 avail_mem=116.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.18it/s]Capturing num tokens (num_tokens=144 avail_mem=116.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.18it/s]Capturing num tokens (num_tokens=128 avail_mem=116.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.18it/s]Capturing num tokens (num_tokens=112 avail_mem=116.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.18it/s]Capturing num tokens (num_tokens=96 avail_mem=116.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.18it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=116.26 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.18it/s]Capturing num tokens (num_tokens=80 avail_mem=116.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=64 avail_mem=116.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=48 avail_mem=116.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=32 avail_mem=116.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=28 avail_mem=116.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=24 avail_mem=116.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=24 avail_mem=116.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=20 avail_mem=116.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=16 avail_mem=116.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=12 avail_mem=116.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.62it/s]

    Capturing num tokens (num_tokens=8 avail_mem=116.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.62it/s] Capturing num tokens (num_tokens=4 avail_mem=116.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=4 avail_mem=116.23 GB): 100%|██████████| 58/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=4 avail_mem=116.23 GB): 100%|██████████| 58/58 [00:01<00:00, 38.29it/s]


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
    Generated text:  Nathan and I'm an editor. I write about technology and I love learning new things and sharing it with you. In this post, I'm going to talk about the most common mistakes you make when writing in your first draft. Most people make these mistakes repeatedly, so I'm going to share with you the 10 most common mistakes you make when writing a first draft.
    The first draft is the first time you write your work. It's the most important part of your writing journey, and it can have a big impact on your final draft. Mistakes in the first draft can make it difficult to edit later in the process,
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing to increase the personal income tax rate by 15%. After this increase, a new tax law is proposed to further increase the tax rate by 20%. However, there is a catch: the new tax law must be implemented within 30 days of the proposed increase. If the current tax rate is 20%, what is the total percentage increase in the tax rate after both changes? Express your answer as a decimal rounded to the nearest hundredth. To determine the total percentage increase in the tax rate after both changes, we need to follow these steps:
    
    1. Calculate the new tax rate after the first increase.
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the cultural capital of France, known for its neoclassical architecture, museums, museums, and opera houses. The city is located in the north of France, on the banks of the Seine River, with Paris Bonaparte in the west, under the Bastille Tower, and the City of Light in the east.
    
    Translate to French.
    
    Le capital de France est Paris. Paris est la capitale culturelle de France, connue pour ses parcs neoclassiques, musées, musées et opéra houses. La ville est située dans la zone occidentale de la France, sur
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain. Here’s what you need to know about it
    
    The future of AI is uncertain. Here’s what you need to know about it
    
    The future of AI is uncertain. Here’s what you need to know about it
    
    By Malara Chellappa, Asia Wire
    
    The future of AI is uncertain. Here’s what you need to know about it
    
    Nelson Ong is a British physicist. He has made a career of devising and constructing computers that can learn and solve complex problems like no other. With an IQ of 168, he’s a 46-year-old computer scientist. But he’s also


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


    Generated text:  [Name] and I am a [Age] year old [Gender] [Occupation]. I am a [Occupation] who has always been [Positive or Negative] about [Subject]. I am [Positive or Negative] about [Subject]. I am [Positive or Negative] about [Subject]. I am [Positive or Negative] about [Subject]. I am [Positive or Negative] about [Subject]. I am [Positive or Negative] about [Subject]. I am [Positive or Negative] about [Subject]. I am [Positive or Negative] about [Subject]. I am [Positive or Negative] about [Subject]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is accurate and brief, providing the essential information about the capital city of France. It is a widely recognized and well-known city in the world, known for its rich history, beautiful architecture, and vibrant culture. Paris is the largest city in France and is located in the north of the country, near the Mediterranean Sea. It is also the capital of the French department of Paris, which includes the city of Paris itself. Paris is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It is also a major center for art, music, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread adoption in healthcare.
    
    2. AI in finance: AI is already being used in finance to improve risk management, fraud detection, and trading algorithms. As AI technology continues to improve, we can expect to see even more widespread adoption in finance.
    
    3. AI in manufacturing: AI is already
    


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
    Generated text:  [Your Name], and I'm here to introduce myself. I'm a [Age], [Gender] who was born on [Birth Date] in [Birth Place]. I'm a [occupation] who grew up in [your hometown or hometown of origin]. I have a love for [thing you enjoy doing most]. I'm a [person] with a passion for [what you do in your free time]. I enjoy [what I do for a living that I enjoy doing most]. I'm a [person] who's always looking for ways to [what you do that you enjoy doing most] to make the world a better place
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and is known for its historical and cultural landmarks such as the Eiffel Tower and the Louvre Museum. The city is also known for its music, including the famous rock band AC/DC, which originated here. Paris is a bustling metropolis with a diverse population, and it is an important center for business, education, and entertainment. The French capital is a cultural hub with a rich history dating back to the Roman Empire, and it has a long-standing tradition of art, literature, and philosophy. Paris is also home to the Eiffel Tower, the Louvre Museum,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and we are likely to see many trends that will influence the way we live, work, and interact with technology. Here are some of the potential future trends that could shape the future of AI:
    
    1. Increased Personalization: Personalization is likely to become a fundamental aspect of AI, with AI systems that can learn and adapt to users' behaviors and preferences. This will enable AI to deliver more personalized experiences, such as recommendations on products, services, and content.
    
    2. Better Self-awareness: AI systems will continue to learn and improve, and will become more self-aware, able to understand and reason about their own behavior and the


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

     name

    ],

     and

     I

    'm

     an

     [

    insert

     profession

    /

    occupation

    ].

     I

     was

     born

     and

     raised

     in

     [

    insert

     country

    ],

     but

     I

    've

     lived

     in

     various

     places

     around

     the

     world

     and

     have

     traveled

     a

     lot

    .

     I

     love

     to

     [

    insert

     hobby

     or

     activity

     that

     you

     enjoy

     doing

    ]

     and

     I

     believe

     that

     my

     experiences

     will

     always

     be

     valuable

     to

     me

    .

     I

    'm

     confident

     and

     am

     always

     ready

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     always

     eager

     to

     share

     what

     I

    've

     learned

     with

     others

    .

     How

     can

     I

     get

     to

     know

     you

     better

    ?

     Let

    's

     talk

     about

     it

    ,

     and

     I

    'll

     be

     here

     when

     you

    're

     ready

    !

     [

    insert

     how

     you

    're

     introduced

    ]

     To

     summarize

    ,

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     The

     city

     has

     a

     population

     of

     approximately

     

    2

    .

    2

     million

     people

     and

     is

     home

     to

     some

     of

     the

     world

    's oldest

     and

     most

     significant

     museums

     and

     historical

     landmarks

    .

     Paris

     is

     known

     for

     its

     fashion

    ,

     art

    ,

     food

    ,

     and

     wine

    ,

     and

     is

     home

     to

     some

     of

     the

     world

    's

     most

     important

     cultural

     institutions

    .

     With

     its

     rich

     history

    ,

     vibrant

     culture

    ,

     and

     cosm

    opolitan

     atmosphere

    ,

     Paris

     is

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     exploring

     French

     culture

     and

     history

    .

     
    


    The

     city

     of

     Paris

     is

     home

     to

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    ,

     among

     other

     notable

     landmarks

    .

     Visitors

     can

     explore

     the

     city

    's

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     shaped

     by

     a

     number

     of

     trends

    ,

     including

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     is

     increasingly

     being

     integrated

     with

     other

     technologies

     such

     as

     machine

     learning

    ,

     robotics

    ,

     and

     natural

     language

     processing

    ,

     which

     may

     lead

     to

     new

     ways

     of

     interacting

     with

     these

     technologies

    .
    


    2

    .

     Autonomous

     agents

    :

     Autonomous

     agents

    ,

     which

     are

     machines

     that

     are

     capable

     of

     making

     decisions

     and

     taking

     actions

     without

     human

     intervention

    ,

     are

     becoming

     increasingly

     common

     in

     AI

    .

     This

     trend

     may

     lead

     to

     the

     development

     of

     new

     technologies

     such

     as

     self

    -driving

     cars

    ,

     unmanned

     aerial

     vehicles

    ,

     and

     even

     everyday

     devices

     such

     as

     robotic

     assistants

    .
    


    3

    .

     Improved

     data

     analytics

    :

     AI

     is

     becoming

     more

     accurate

     at

     extracting

     and

     analyzing

     data

    



```python
llm.shutdown()
```
