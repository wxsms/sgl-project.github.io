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
    [2026-04-23 07:26:23] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.95it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.93it/s]


    2026-04-23 07:26:28,125 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 07:26:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.24it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.24it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.24it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.24it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.24it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:23,  2.24it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.24it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:23,  2.24it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:07,  6.54it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:07,  6.54it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:07,  6.54it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:07,  6.54it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:07,  6.54it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:07,  6.54it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:07,  6.54it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:07,  6.54it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:07,  6.54it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:07,  6.54it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.40it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.40it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.40it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.40it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.40it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.40it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.40it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.40it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.40it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.40it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.30it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.30it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.30it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.30it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.30it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.30it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.30it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.30it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.30it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.30it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.99it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.99it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.99it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.99it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.99it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.99it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.99it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 29.99it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 29.99it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 29.99it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.13it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.13it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.13it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.68it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  31%|███       | 18/58 [00:00<00:01, 34.36it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 34.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 34.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  31%|███       | 18/58 [00:00<00:01, 34.36it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  31%|███       | 18/58 [00:00<00:01, 34.36it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  31%|███       | 18/58 [00:00<00:01, 34.36it/s]Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.08it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.08it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.08it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.08it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.08it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.08it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.80it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.80it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.80it/s]

    Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.80it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.80it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.80it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.08it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.08it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.08it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.08it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.08it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.08it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.83it/s]

    Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.53it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.53it/s]

    Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.07it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 38.01it/s]


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
    Generated text:  Michael John Miller. I am 27 years old. I'm a maths student at Liverpool University. I'm looking forward to some of the events on the Mathematical Olympiad website. In the process of solving the problems, I always think about the beauty and simplicity of mathematics. This is because I have a very strong interest in numbers, and I believe that it's a wonderful way to learn mathematics. I'm also quite good at problem-solving. For instance, one of the problems that I often think about is whether it's true that if the square of a number is equal to the number itself. I don't mean to say that
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking officer of the armed forces. The president has the authority to appoint and remove government officials, to sign or veto legislation, to decide how to handle issues of national security and foreign policy. President Obama appointed Hillary Clinton to the presidency, and now the president has lost the presidential election by 2 points. You are a fan of the football team the Washington Redskins. You have decided to participate in a game of football. The quarterback of the Washington Redskins is named Shaun Rogers. He is the leader of the Redskins football team. The Redskins football team has the winning record of 12 wins and 2 losses. The Redskins are
    ===============================
    Prompt: The capital of France is
    Generated text:  _____. A: Paris B: Lyon C: Nice D: Marseille
    Answer: To determine the correct answer, let's analyze each option step by step:
    
    A: Paris - This is the capital of France, and it is the largest city in the country.
    
    B: Lyon - Lyon is the capital of France but is not the largest city. It is located on the Loire river and is part of the Lyon metropolitan area.
    
    C: Nice - Nice is the capital of France but is not the largest city. It is a coastal city and is part of the Nice metropolitan area.
    
    D: Marseille - Marseille is the capital of France
    ===============================
    Prompt: The future of AI is
    Generated text:  shaping the way we interact with technology. In this blog post, we will explore how AI is transforming the way we interact with technology, including its impact on healthcare, finance, and education.
    The future of AI is shaping the way we interact with technology, including its impact on healthcare, finance, and education.
    AI is revolutionizing the way we interact with technology, from healthcare to finance to education. Here are some of the ways in which AI is changing the way we interact with technology:
    AI is transforming healthcare by improving the efficiency and accuracy of medical diagnosis and treatment. AI-powered medical assistants are being used to review medical records, diagnose diseases


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and a vibrant culture. It is located in the south of France and is the largest city in the country. Paris is known for its beautiful architecture, world-renowned museums, and annual festivals such as the Eiffel Tower and the Louvre. The city is also famous for its food, fashion, and music. Paris is a popular tourist destination and a cultural hub for France. It is home to many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its annual
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve efficiency. As AI becomes more advanced, it is likely to be used in more complex and personalized ways, with the potential to revolutionize the field of medicine.
    
    3. Greater use of AI in finance: AI is already being used in finance to improve fraud
    


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
    Generated text:  [Name] and I'm a [Age] year old [Gender] person. I love [career], [skill], and [interest] [and I enjoy spending my free time [activities, hobbies, etc.]]. I'm [skin color], [ethnicity], and [religion]. I'm [occupation] and I'm [level of perfection]. I'm confident in [my abilities, skills, or traits] and [I'm passionate about [sports, hobbies, etc.]]. I'm [born in [location]], [religion] [and I have a strong [attitude] towards [career, hobby,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the capital of France and is the largest city in the country, with a population of over 2. 5 million people. It is a city located on the River Seine, on the left bank of the river, in the central region of the country. Paris is known for its rich history, culture, and art, and it is home to many famous landmarks and attractions such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city is also a major economic center and a major hub of the French media and entertainment industry. Paris is known as the "City of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating, and there are many possibilities for how it will evolve. Here are some of the most promising trends that could shape the way we interact with AI in the years to come:
    
    1. Autonomous and Human-AI Collaboration: One of the most promising areas for AI development is the development of a technology that could merge human and AI capabilities. This technology could allow humans to work alongside AI to perform tasks that require human expertise, such as decision-making, creativity, and problem-solving. It could also enable more personalized and efficient healthcare solutions.
    
    2. Augmented and Virtual Reality: As VR technology continues to advance, there's a high likelihood that


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

    Character

     Name

    ]

     and

     I

    'm

     a

    /an

     [

    occupation

    ]

     with

     over

     [

    number

     of

     years

    ]

     years

     in

     [

    profession

    ].

     My

     love

     for

     [

    job

    ]

     is

     so

     strong

     that

     I

    've

     been

     [

    number

     of

     years

    ]

     years

     focused

     on

     it

    .

     I

    've

     [

    number

     of

     books

    ,

     articles

    ,

     or

     other

     resources

    ]

     learned

     about

     [

    job

    ],

     and

     [

    number

     of

     chapters

     in

     a

     book

    ,

     articles

     in

     an

     article

    ,

     or

     other

     resource

    ]

     have

     also

     taught

     me

     how

     to

     be

     [

    job

    ].

     My

     strength

     is

     my

     passion

     for

     [

    job

    ]

     and

     my

     unw

    av

    ering

     commitment

     to

     it

    .

     I

     have

     a

     [

    number

     of

     friends

    ,

     mentors

    ,

     or

     co

    -workers

    ]

     who

     support

     me

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

     and

     is

     the

     capital

     of

     the

     country

    .

     It

     is

     home

     to

     many

     famous

     historical

     sites

    ,

     cultural

     landmarks

    ,

     and

     iconic

     buildings

    .

     The

     city

     is

     also

     known

     for

     its

     lively

     nightlife

     and

     vibrant

     culture

    .

     Paris

     is

     a

     hub

     of

     art

    ,

     music

    ,

     and

     fashion

    ,

     and

     has

     been

     recognized

     as

     one

     of

     the

     world

    's

     most

     liv

    able

     cities

    .

     The

     city

    's

     architecture

    ,

     including

     the

     iconic

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

    ,

     are

     some

     of

     the

     world

    's

     most

     recognizable

     landmarks

    .

     The

     French

     government

     has

     implemented

     policies

     to

     preserve

     and

     promote

     the

     city

    's

     cultural

     heritage

    ,

     making

     it

     a

     UNESCO

     World

     Heritage

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     with

     many

     potential

     developments

     and

     possibilities

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     accuracy

    :

     With

     the

     development

     of

     deep

     learning

     and

     neural

     networks

    ,

     AI

     systems

     are

     becoming

     more

     accurate

     and

     precise

    .

     This

     will

     lead

     to

     more

     reliable

     and

     trustworthy

     AI

     applications

    .
    


    2

    .

     Automation

    :

     As

     AI

     becomes

     more

     capable

    ,

     it

     will

     be

     able

     to

     automate

     tasks

     that

     would

     previously

     require

     human

     intervention

    .

     This

     could

     lead

     to

     increased

     efficiency

     and

     productivity

    ,

     but

     it

     could

     also

     result

     in

     job

     displacement

     for

     some

     workers

    .
    


    3

    .

     AI

     ethics

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     there

     will

     likely

     be

     ethical

     considerations

     surrounding

     their

     use

    .

     This

     could

     include

     issues

     such

     as

     bias

    ,

    



```python
llm.shutdown()
```
