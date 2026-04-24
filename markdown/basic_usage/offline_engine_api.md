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
    [2026-04-24 06:26:00] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.89it/s]


    2026-04-24 06:26:05,046 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 06:26:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:02<00:11,  4.30it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:02<00:11,  4.30it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:02<00:11,  4.30it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:11,  4.30it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:03<00:11,  4.30it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:03<00:11,  4.30it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:03<00:11,  4.30it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:04,  8.97it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:04,  8.97it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:04,  8.97it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:04,  8.97it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:04,  8.97it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:03<00:04,  8.97it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:03<00:04,  8.97it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:03<00:04,  8.97it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 15.52it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 15.52it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 15.52it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 15.52it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 15.52it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 15.52it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 15.52it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 15.52it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 15.52it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 23.75it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 23.75it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 23.75it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 23.75it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 23.75it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 23.75it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 23.75it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 23.75it/s]

    Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 23.75it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 32.22it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 32.22it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 32.22it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 32.22it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 32.22it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 32.22it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 32.22it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 32.22it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 32.22it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 40.38it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 40.38it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 40.38it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 40.38it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 40.38it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 40.38it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 40.38it/s]

    Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 40.38it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 40.38it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 47.54it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 47.54it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 47.54it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 47.54it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 47.54it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 47.54it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.35 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.35 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.35 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.35 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.35 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.34 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.34 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.34 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.34 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=116.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.32 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.32 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.32 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.32 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.31 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.31 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.31 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.31 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.30 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.28 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s]

    Capturing num tokens (num_tokens=960 avail_mem=116.29 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s] Capturing num tokens (num_tokens=896 avail_mem=116.29 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=896 avail_mem=116.29 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=832 avail_mem=116.29 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=768 avail_mem=116.29 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=704 avail_mem=116.28 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=640 avail_mem=116.28 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=576 avail_mem=116.28 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=576 avail_mem=116.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.72it/s]Capturing num tokens (num_tokens=512 avail_mem=116.27 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.72it/s]Capturing num tokens (num_tokens=480 avail_mem=116.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.72it/s]

    Capturing num tokens (num_tokens=448 avail_mem=116.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.72it/s]Capturing num tokens (num_tokens=416 avail_mem=116.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.72it/s]Capturing num tokens (num_tokens=384 avail_mem=116.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.72it/s]Capturing num tokens (num_tokens=384 avail_mem=116.28 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=352 avail_mem=116.27 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=320 avail_mem=116.27 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=288 avail_mem=116.26 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=256 avail_mem=116.26 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=240 avail_mem=116.26 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.70it/s]Capturing num tokens (num_tokens=240 avail_mem=116.26 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=224 avail_mem=116.26 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]

    Capturing num tokens (num_tokens=208 avail_mem=116.25 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=192 avail_mem=116.25 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=176 avail_mem=116.25 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=160 avail_mem=116.24 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=160 avail_mem=116.24 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=144 avail_mem=116.24 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=128 avail_mem=116.24 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=112 avail_mem=116.24 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=96 avail_mem=116.23 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.30it/s] Capturing num tokens (num_tokens=80 avail_mem=116.23 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.30it/s]

    Capturing num tokens (num_tokens=80 avail_mem=116.23 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=64 avail_mem=116.22 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=48 avail_mem=116.22 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=32 avail_mem=116.22 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=28 avail_mem=116.21 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=24 avail_mem=116.21 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=24 avail_mem=116.21 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.83it/s]Capturing num tokens (num_tokens=20 avail_mem=116.21 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.83it/s]Capturing num tokens (num_tokens=16 avail_mem=116.14 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.83it/s]Capturing num tokens (num_tokens=12 avail_mem=116.13 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.83it/s]Capturing num tokens (num_tokens=8 avail_mem=116.13 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.83it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=116.12 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.83it/s]Capturing num tokens (num_tokens=4 avail_mem=116.12 GB): 100%|██████████| 58/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=4 avail_mem=116.12 GB): 100%|██████████| 58/58 [00:01<00:00, 38.58it/s]


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
    Generated text:  Kristina. I have been writing poetry for many years now. My main focus is on the world of the absurd, the unconventional, and the unexpected. My goal is to explore the human condition in the light of a wry and sometimes humorous lens.
    I come from a family of visionaries and do not shy away from the wild. I have published my poems on a number of online platforms. I am happy to share my work with you.
    I'm a professional graphic designer and I used to work for a big advertising agency. It was a time when I knew all the tricks to create the most stunning visual effect possible. I do
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 36 years old. How old will the president be in 7 years?
    
    To determine how old the president of the United States will be in 7 years, we start with the current age of the president and add 7 years to it. The current age of the president is 36 years.
    
    Here's the step-by-step calculation:
    
    1. Identify the current age of the president: 36 years.
    2. Add 7 years to the current age: \(36 + 7 = 43\).
    
    Therefore, in 7 years, the president will be \(\boxed{43}\
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population of Paris is 2.1 million.
    A. 正确
    B. 错误
    Answer: B
    
    The Republic of the United States is an officially United States of America, located on the northwestern coast of North America, adjacent to Canada to the west.
    A. 正确
    B. 错误
    Answer: B
    
    In 2014, the number of people in the European Union was about 48 million.
    A. 正确
    B. 错误
    Answer: B
    
    1.0.2.0.0 is the default IP address
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but we must think carefully about how we use it.
    - We need to be mindful of the risks that AI poses to human life, intelligence, and well-being.
    - We must consider the long-term consequences of AI development and deployment.
    - We must work together to ensure that the benefits of AI outweigh the risks and that it is used for the common good.
    - We must prioritize ethical considerations in the development, deployment, and use of AI.
    - We must promote transparency, accountability, and trust in the AI ecosystem.
    - We must respect the rights of all people, including those with disabilities, and ensure that AI is accessible and


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. Let's chat! [Name] [Company Name] is a [brief description of the company]. I'm [age] years old and I'm [job title]. I'm always looking for new challenges and opportunities to grow and learn. What can I expect from our conversation? [Name] [Company Name] is a [brief description of the company]. I'm [age] years old and I'm [job title]. I'm always looking for new challenges and opportunities to grow
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its fashion industry and its role in the French Revolution. It is the largest city in France and is a major economic and political center. Paris is a city of contrasts, with its historical architecture and modern architecture blending together to create a unique and fascinating city. The city is also home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries. This will lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI-powered healthcare: AI will play a crucial role in healthcare, with the ability to analyze large amounts of data and provide personalized treatment plans. This will lead to improved patient outcomes and reduced costs.
    
    3. AI-powered education: AI will
    


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
    Generated text:  [Name], and I'm a [职业/年龄] [Age] [Name] from [Location]. I'm a versatile and adaptable individual who thrives in many different roles, and I love to learn new things and expand my skillset. I enjoy challenging myself to constantly improve and pursue my passions in life. I'm a friendly and approachable person, and I'm always ready to share my knowledge and insights with others. My personality is infectious and I love sharing my enthusiasm for the things I love. I'm passionate about helping others achieve their goals, and I believe in the power of making a positive impact on the world.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known as the "City of Love" due to its rich cultural heritage and romantic atmosphere. Paris has a vibrant nightlife and numerous attractions, including the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. The city is also home to numerous museums, such as the Musée de l'Orangerie and the Musée d'Orsay, which showcase a variety of artistic and historical pieces. Paris is known for its long hours of daylight and its unique French accent, which makes it a popular tourist destination.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see significant advancements in areas such as:
    
    1. Improved accuracy and precision in natural language processing: With advances in machine learning, it is expected that AI systems will be able to understand and interpret human language more accurately, leading to more natural and conversational interactions.
    
    2. Enhanced security and privacy: AI systems will continue to become more sophisticated and capable of identifying and protecting against various types of cyber threats, such as malware and ransomware.
    
    3. Autonomous vehicles: Autonomous vehicles are likely to become more prevalent in the future, as AI technologies advance to the point where they can be used to safely and effectively navigate and operate in complex and


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

     Sarah

     and

     I

    'm

     a

     dedicated

     journalist

    .

     I

    've

     always

     been

     fascinated

     by

     the

     stories

     of

     people

     around

     the

     world

     and

     have

     a

     knack

     for

     turning

     a

     seemingly

     ordinary

     day

     into

     a

     riv

    eting

     piece

     of

     reporting

    .

     Whether

     it

    's

     interviewing

     famous

     figures

     or

     chronic

    ling

     everyday

     struggles

    ,

     my

     passion

     for

     the

     written

     word

     has

     driven

     me

     to

     constantly

     hone

     my

     skills

     and

     seek

     out

     new

     and

     interesting

     stories

    .

     As

     a

     journalist

    ,

     I

     strive

     to

     uncover

     the

     truth

     and

     bring

     it

     to

     light

    ,

     and

     I

    'm

     constantly

     learning

     and

     improving

     my

     craft

    .

     I

    'm

     excited

     to

     meet

     you

     and

     get

     to

     know

     you

    ,

     Sarah

    .

     Welcome

     to

     my

     world

    .

     How

     are

     you

     feeling

     about

     the

     day

     so

     far

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    .


    Paris

     is

     the

     cultural

     and

     economic

     center

     of

     France

    ,

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     home

     to

     many

     famous

     French

     artists

    ,

     writers

    ,

     and

     musicians

    .

     It

     has

     a

     long

     history

     dating

     back

     to

     the

     Roman

     Empire

    ,

     and

     has

     been

     a

     center

     of

     learning

     and

     culture

     for

     centuries

    .

     Paris

     has

     been

     recognized

     as

     a

     UNESCO

     World

     Heritage

     Site

     several

     times

     and

     is

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    .

     As

     of

     

    2

    0

    2

    1

    ,

     Paris

     has

     a

     population

     of

     approximately

     

    2

    .

     

    6

     million

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     some

     possible

     trends

     are

    :
    


    1

    .

     Increased

     automation

    :

     AI

     systems

     will

     become

     more

     sophisticated

     and

     capable

     of

     performing

     tasks

     that

     were

     previously

     done

     by

     humans

    .

     This

     could

     lead

     to

     a

     shift

     towards

     automation

     of

     tasks

     that

     are

     both

     repetitive

     and

     dangerous

    ,

     such

     as

     factory

     work

     or

     industrial

     manufacturing

    .
    


    2

    .

     Increased

     integration

     of

     AI

     and

     human

     expertise

    :

     AI

     systems

     will

     become

     more

     integrated

     with

     human

     expertise

    ,

     allowing

     for

     more

     efficient

     and

     effective

     collaboration

     between

     humans

     and

     machines

    .
    


    3

    .

     AI

     ethical

     considerations

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     there

     will

     be

     greater

     ethical

     considerations

     around

     their

     use

    .

     This

     includes

     issues

     around

     bias

    ,

     transparency

    ,

     and

     accountability

    .
    


    4

    .

     AI

     disruption

     of

    



```python
llm.shutdown()
```
