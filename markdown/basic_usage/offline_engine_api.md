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


    [transformers] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [transformers] `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    [transformers] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [transformers] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [transformers] `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-27 08:02:23] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.72it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.70it/s]


    2026-04-27 08:02:28,383 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 08:02:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=5632):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=3328):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]

    Compiling num tokens (num_tokens=3072):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:10,  4.35it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:10,  4.35it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:10,  4.35it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:05<00:10,  4.35it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:05<00:10,  4.35it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:05<00:10,  4.35it/s]Compiling num tokens (num_tokens=1024):  22%|██▏       | 13/58 [00:05<00:10,  4.35it/s]Compiling num tokens (num_tokens=960):  22%|██▏       | 13/58 [00:05<00:10,  4.35it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:04,  8.92it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.61it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 21.44it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 28.97it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.68 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.67 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.67 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.67 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.67 GB):   9%|▊         | 5/58 [00:00<00:03, 15.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.66 GB):   9%|▊         | 5/58 [00:00<00:03, 15.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.65 GB):   9%|▊         | 5/58 [00:00<00:03, 15.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.19it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=118.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.64 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.64 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.64 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.63 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.63 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.63 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.63 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.00it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=118.63 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.61 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.60 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.66it/s]Capturing num tokens (num_tokens=960 avail_mem=118.61 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.66it/s] Capturing num tokens (num_tokens=896 avail_mem=118.61 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.66it/s]Capturing num tokens (num_tokens=832 avail_mem=118.60 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.66it/s]

    Capturing num tokens (num_tokens=768 avail_mem=118.60 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.66it/s]Capturing num tokens (num_tokens=768 avail_mem=118.60 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.94it/s]Capturing num tokens (num_tokens=704 avail_mem=118.60 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.94it/s]Capturing num tokens (num_tokens=640 avail_mem=118.59 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.94it/s]Capturing num tokens (num_tokens=576 avail_mem=118.59 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.94it/s]Capturing num tokens (num_tokens=512 avail_mem=118.58 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.94it/s]Capturing num tokens (num_tokens=480 avail_mem=118.59 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.94it/s]Capturing num tokens (num_tokens=480 avail_mem=118.59 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.97it/s]Capturing num tokens (num_tokens=448 avail_mem=118.59 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.97it/s]Capturing num tokens (num_tokens=416 avail_mem=118.59 GB):  52%|█████▏    | 30/58 [00:01<00:00, 36.97it/s]Capturing num tokens (num_tokens=384 avail_mem=118.59 GB):  52%|█████▏    | 30/58 [00:01<00:00, 36.97it/s]

    Capturing num tokens (num_tokens=352 avail_mem=118.58 GB):  52%|█████▏    | 30/58 [00:01<00:00, 36.97it/s]Capturing num tokens (num_tokens=320 avail_mem=118.58 GB):  52%|█████▏    | 30/58 [00:01<00:00, 36.97it/s]Capturing num tokens (num_tokens=320 avail_mem=118.58 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=288 avail_mem=118.57 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=256 avail_mem=118.57 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=240 avail_mem=118.57 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=224 avail_mem=118.56 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=208 avail_mem=118.56 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=208 avail_mem=118.56 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=192 avail_mem=118.56 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=176 avail_mem=118.56 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.24it/s]

    Capturing num tokens (num_tokens=160 avail_mem=118.55 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=144 avail_mem=118.55 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=128 avail_mem=118.55 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=128 avail_mem=118.55 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=112 avail_mem=118.55 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=96 avail_mem=118.54 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.49it/s] Capturing num tokens (num_tokens=80 avail_mem=118.54 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=64 avail_mem=118.53 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=48 avail_mem=118.53 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=48 avail_mem=118.53 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=32 avail_mem=118.53 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.03it/s]

    Capturing num tokens (num_tokens=28 avail_mem=118.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=24 avail_mem=118.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=20 avail_mem=118.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=16 avail_mem=118.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=16 avail_mem=118.52 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=12 avail_mem=118.51 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=8 avail_mem=118.51 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.86it/s] Capturing num tokens (num_tokens=4 avail_mem=118.51 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=4 avail_mem=118.51 GB): 100%|██████████| 58/58 [00:01<00:00, 35.48it/s]


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
    Generated text:  Vova and I am a senior computer science major. I have taken many different computer science courses. I am studying computer systems and systems architecture, as well as the design of algorithms, data structures, and software engineering. I have taken many programming languages and courses in them.
    
    What are the courses that I have taken and what are some of the topics in the courses that are relevant to computer science?
    
    I took several computer science courses. Here are some of the courses I have taken:
    
      1. Introduction to Computer Science and Programming - this course helped me understand the basics of computer science and programming, such as algorithms, data structures,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful figure in the United States. The president is responsible for the administration and the government of the United States. The president of the United States serves a five-year term. The vice president of the United States serves as the president’s choice to fill the vacancy in the office when the president retires. The president also serves as the head of the United States military and serves in a position of great power.
    While the president is the most powerful person in the United States, the vice president is in a similar role. The vice president does not have to be as powerful as the president, but he or she is still a leader in the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. In the city of Paris, there is a park that is 150 meters wide. This park has a 15-meter wide path around its perimeter, making it 165 meters wide in total. If you start from the edge of the park and walk 20 meters to the edge of the park, how many meters is the remaining distance from the edge of the park to the center of the park?
    
    To determine the remaining distance from the edge of the park to the center of the park, we need to follow these steps:
    
    1. **Calculate the total width of the park including the path:**
    
    ===============================
    Prompt: The future of AI is
    Generated text:  about creating, improving and expanding artificial intelligence. When we talk about artificial intelligence, we talk about AI, big and small. Big AI is the autonomous AI that is developed for autonomous systems that are complex in terms of resources and computational power. These systems include robotics, AI driven traffic light systems, autonomous vehicles, and many others. Small AI is the small, low-powered AI that is being developed and used for various applications, such as in banking, security, and medical. In this article, we will discuss the future of AI in banks and how it will be transformed by different factors such as automation, big data, and innovation. Let


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, art, and culture, and is a major tourist destination. The city is home to many famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for business, finance, and education. The city is home to many important institutions and organizations, including the French Academy of Sciences and the French National Library. Paris is a vibrant and dynamic city with a rich cultural and historical heritage. The city is also known for its cuisine, including
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical AI: As more and more AI systems become involved in decision-making processes, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and fairness.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, transportation, and manufacturing. As more and more of these technologies become integrated with
    


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
    Generated text:  [Name] and I'm [Name]!
    
    I'm a/an [Job Title] with [Number of Years] years of experience in the [Industry] industry. I have a strong passion for [Professional Interest or Area of Expertise]. I'm always looking for opportunities to grow and learn, and I enjoy [Professional Interest or Area of Expertise].
    
    I'm also [Other Professional Interest or Area of Expertise]. I'm a hard worker who is always committed to my goals and responsible for achieving them. I love being a part of a team and I'm always looking for ways to improve my skills and knowledge.
    
    In my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that is known for its grand boulevards, its rich history, and its iconic Eiffel Tower. With its long, winding streets and picturesque landmarks, Paris is a popular tourist destination that attracts millions of visitors each year. The city is also home to various museums, theaters, and cultural institutions, making it a must-visit destination for anyone interested in French culture and history. Visitors to Paris can explore the vibrant music scene, enjoy the world-famous Eiffel Tower, and learn about French cuisine, art, and architecture through its numerous cultural institutions. Paris is a city that has it all and offers something
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a number of different trends, including:
    
    1. Increased integration with existing technologies: As AI becomes more integrated into existing technologies, such as smartphones, smart homes, and smart cities, it is likely that we will see even more seamless and integrated AI systems that seamlessly integrate with various other technologies.
    
    2. Increased focus on ethical AI: As more and more AI systems become involved in decision-making, it is likely that there will be an increased focus on ethical AI, ensuring that AI systems are designed and used in a way that is fair, transparent, and respectful of human rights and values.
    
    3. Advancements in AI


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

    ]

     and

     I

     am

     [

    occupation

    ].

     I

     am

     [

    number

    ]

     years

     old

    .

     I

     enjoy

     [

    reason

    s

     why

     I

     love

     to

     travel

    ]

     and

     I

     love

     [

    activities

     I

     like

     to

     do

    ].

     I

     also

     like

     [reason

     why

     I

     like

     to

     eat

    ].

     My favorite

     place

     to

     eat

     is

     [

    name

     of

     restaurant

    ].

     I

     can

     speak

     [

    language

    ] and

     I

     have

     been

     studying

     [

    skill

    ]

     for

     [

    number

    ]

     years

    .

     I

     travel

     a

     lot

     and

     often

     I

     eat

     at

     my

     favorite

     restaurant

    ,

     so

     I

     am

     eager

     to

     learn

     about

     the

     food

    .

     I

     also

     like

     to

     write

     a

     blog

     about

     my

     travels

     and

     experiences

    .

     I

     have

     a

     very

     positive

     and

     energetic

     personality

    .

     My

     hobbies

     include

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     country

     and

     home

     to

     many

     of

     its

     most

     famous

     landmarks

     and

     attractions

    .

     The

     city

     is

     also

     the

     seat

     of

     the

     government

     and

     administrative

     center

     of

     France

    .

     It

     is

     a

     popular

     tourist

     destination

     for

     tourists

     and

     locals

     alike

    .

     Paris

     is

     known

     for

     its

     architecture

    ,

     vibrant

     culture

    ,

     and

     rich

     history

    .

     The

     city

     is

     also

     home

     to

     many

     notable

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

     d

    '

    Or

    say

    .

     Paris

     is

     known

     for

     its

     cuisine

    ,

     including

     the

     famous

     French

     fries

    ,

     and

     its

     social

     events

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Carn

    aval

     de

     Paris

    .

     It

     is

     a

     city

     of

     contrasts

     and

     has

     been

     a

     cultural

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     rapid

     growth

     in

     the

     complexity

     and

     sophistication

     of

     AI

     systems

    ,

     as

     well

     as

     a

     growing

     emphasis

     on

     ethical

     considerations

     and

     accountability

    .

     Here

     are

     some

     potential

     trends

     to

     consider

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

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

     a

     greater

     emphasis

     on

     ethical

     considerations

    ,

     including

     issues

     such

     as

     bias

    ,

     accountability

    ,

     and

     transparency

    .

     This

     will

     likely

     involve

     a

     shift

     in

     the

     way

     we

     design

     and

     deploy

     AI

     systems

    ,

     and

     a

     greater

     willingness

     to

     engage

     with

     the

     ethical

     implications

     of

     AI

    .
    


    2

    .

     Greater

     integration

     with

     other

     technologies

    :

     As

     AI

     systems

     become

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

    ,

     wear

    ables

    ,

     and

    



```python
llm.shutdown()
```
