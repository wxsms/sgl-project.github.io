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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-16 22:57:10] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.36it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.34it/s]


    2026-04-16 22:57:14,827 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 22:57:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:03<00:07,  6.19it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 13.38it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 13.38it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 13.38it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 13.38it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.38it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.38it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.38it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 13.38it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 13.38it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 13.38it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 21.57it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 21.57it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 21.57it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 21.57it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 21.57it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 21.57it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 21.57it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 21.57it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 21.57it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 21.57it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 30.53it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 30.53it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 39.70it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 39.70it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 39.70it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 39.70it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 39.70it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 39.70it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 39.70it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 39.70it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 39.70it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 39.70it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 48.87it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 48.87it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 48.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.71 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.43 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.43 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=121.43 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.71 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.71 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.70 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.70 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.70 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.70 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.70 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.69 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.69 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.69 GB):  31%|███       | 18/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.69 GB):  31%|███       | 18/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.68 GB):  31%|███       | 18/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.66 GB):  31%|███       | 18/58 [00:00<00:01, 33.62it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.68 GB):  31%|███       | 18/58 [00:00<00:01, 33.62it/s] Capturing num tokens (num_tokens=896 avail_mem=120.34 GB):  31%|███       | 18/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=896 avail_mem=120.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.06it/s]

    Capturing num tokens (num_tokens=448 avail_mem=120.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.18it/s]Capturing num tokens (num_tokens=352 avail_mem=120.23 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.18it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.18it/s]Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=256 avail_mem=120.22 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=256 avail_mem=120.22 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.38it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.38it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.38it/s]

    Capturing num tokens (num_tokens=208 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.38it/s]Capturing num tokens (num_tokens=192 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.38it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.38it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=96 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.36it/s] Capturing num tokens (num_tokens=96 avail_mem=120.19 GB):  81%|████████  | 47/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 41.18it/s]

    Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=48 avail_mem=120.18 GB):  81%|████████  | 47/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  81%|████████  | 47/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=28 avail_mem=120.17 GB):  81%|████████  | 47/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=28 avail_mem=120.17 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.72it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 37.47it/s]


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
    Generated text:  Caihua and I am a full-time student at the University of Science and Technology of China. I’m a second-year Computer Science major. I’m interested in computer science and related fields, and I'm currently working on a project that involves deep learning. I have been studying Python for the last two years, and I love the syntax and the tools that Python provides. I have also tried Java and C++, and I feel like I have a solid foundation in these languages. I'm a bit of a lazy person, and I'm not really interested in programming. I enjoy watching documentaries and reading science fiction books. I'm looking for
    ===============================
    Prompt: The president of the United States is
    Generated text:  54 years older than 3 times the age of his son. If the president is currently 36 years old, what is the current age of the son?
    Let's denote the age of the son as \( x \). According to the problem, the president is 54 years older than 3 times the age of his son. We can express this relationship with the following equation:
    
    \[ \text{President's age} = 3 \times (\text{Son's age}) + 54 \]
    
    We know the president's current age is 36 years, so we can substitute this value into the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 正确
    B. 错误
    Answer:
    B
    
    A fire truck is a vehicle used to provide fire fighting services. They are classified into three types based on the nature of the fire they fight. What are they? ____ 
    A. Rescue Fire Truck
    B. Fire Fighting Fire Truck
    C. Ambulance Truck
    D. Combination Fire Truck
    Answer:
    ABD
    
    The phase of the AC voltage that has a higher peak value is called the highest (peak) voltage. True or False?
    A. True
    B. False
    Answer:
    A
    
    The "People's Republic of China Safety
    ===============================
    Prompt: The future of AI is
    Generated text:  changing and the need for hands-on experience to truly understand how to make use of it will only become more apparent as time goes on. The more applications of AI are made, the more opportunities there are to gain hands-on experience. You can get an insight into AI by building AI models, using Python libraries, and more.
    In this blog, we will be looking at how to get a hands-on experience with the Python programming language and how to use it to build AI models. In the future, this will allow you to understand how to use Python to build AI models, and gain practical knowledge about using AI in the real world.
    The


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I have a [job title] at [company name]. I'm a [job title] at [company name]. I'm a [job title] at [company name]. I'm a [job title] at [company name]. I'm a [job title] at [company name]. I'm a [job title] at [company name]. I'm a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is the second-largest city in France and has a population of over 2.5 million people. The city is known for its cuisine, fashion, and art, and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its historical architecture and modern art, and is a UNESCO World Heritage site. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to become more capable of learning from large amounts of data, allowing machines to perform tasks that were previously impossible or difficult to achieve. This could lead to more efficient and effective systems that can solve complex problems more quickly and accurately.
    
    3. Greater emphasis on ethical considerations: As
    


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
    Generated text:  [name] and I'm a [job title] at [company name]. I've always had a strong passion for [interest or hobby] and I've been passionate about it since [insert when you started]. Over the years, I've developed a unique and innovative approach to [interest or hobby] that sets me apart from my peers. I'm always looking for new challenges and ideas to explore, and I'm always eager to learn more about [interest or hobby]. I'm looking forward to working with you, and I'm here to help you succeed in your [interest or hobby]. 
    Remember, we're all here to learn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that is synonymous with luxury and culture, with its iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and Louvre Museum. It is a city that has been a significant center of French history, culture, and politics, with a rich history dating back over 1,000 years. The city is also known for its world-renowned fashion industry, with its iconic fashion shops, boutiques, and couture tailoring. Paris is a fascinating city that has much to offer to visitors of all ages and interests. With its blend of traditional and modern culture, rich history, and stunning views
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly exciting, as it continues to evolve at an unprecedented rate. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased focus on ethical AI: As more and more people become aware of the negative impact of AI on society, there is a growing push for ethical and responsible development of AI systems. Governments, corporations, and non-profits are beginning to take a more active role in regulating AI development, making it more transparent and accountable.
    
    2. Increased use of AI in healthcare: With the prevalence of chronic diseases and aging populations, AI is becoming increasingly important in healthcare. AI can help detect diseases and track


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

     [

    Age

    ]

     year

     old

     [

    Gender

    ].

     I

     have

     always

     been

     an

     avid

     reader

     of

     fantasy

     novels

     and

     believe

     that

     my

     love

     for

     books

     has

     shaped

     me

     into

     the

     person

     I

     am

     today

    .

     I

     love

     to

     travel

    ,

     explore

     different

     cultures

     and

     languages

    ,

     and

     have

     a

     passion

     for

     history

     and

     literature

    .

     I

    'm

     always

     looking

     for

     new

     ideas

     and

     have

     a

     keen

     eye

     for

     detail

    .

     I

     enjoy

     meeting

     new

     people

     and

     spending

     time

     with

     them

    ,

     and

     I

    'm

     always

     ready

     to

     learn

     and

     grow

    .

     I

     hope

     to

     meet

     you

     and

     help

     you

     on

     your

     journey

     into

     the

     world

     of

     books

    .

     [

    Name

    ]

     

    📚

    📚

    📚

    📚

    📚

    📚

    📚

    📚

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     iconic

     landmarks

     like

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

     Lou

    vre

     Museum

    .

     It

     is

     also

     one

     of

     the

     world

    's

     most

     cosm

    opolitan

     cities

    ,

     home

     to

     a

     diverse

     population

     and

     a

     rich

     cultural

     heritage

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     with

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

     also

     home

     to

     notable

     museums

    ,

     theaters

    ,

     and

     food

     establishments

     that

     make

     it

     a

     popular

     destination

     for

     entertainment

     and

     shopping

    .

     In

     addition

     to

     its

     political

     and

     historical

     significance

    ,

     Paris

     is

     also

     a

     vibrant

     hub

     for

     art

    ,

     music

    ,

     and

     fashion

    ,

     making

     it

     a

     unique

     and

     influential

     cultural

     center

     of

     France

    .

     
    


    Therefore

    ,

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     one

     of

     rapid

     change

     and

     innovation

    ,

     with

     potential

     to

     transform

     many

     aspects

     of

     our

     lives

     and

     work

    .

     Here

     are

     some

     possible

     trends

     that

     could

     emerge

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     improve

     patient

     outcomes

     and

     reduce

     healthcare

     costs

    .

     As

     more

     data

     is

     collected

     and

     analyzed

     by

     AI

    ,

     it

     will

     become

     more

     accurate

     and

     reliable

    ,

     enabling

     doctors

     to

     make

     more

     informed

     decisions

    .

     AI

     will

     also

     enable

     personalized

     medicine

    ,

     where

     treatments

     are

     tailored

     to

     individual

     patient

     needs

    ,

     leading

     to

     better

     patient

     outcomes

     and

     reduced

     healthcare

     costs

    .
    


    2

    .

     AI

     in

     transportation

    :

     AI

     is

     already

     being

     used

     to

     reduce

     traffic

     congestion

     and

     improve

     transportation

     efficiency

    .

     For

    



```python
llm.shutdown()
```
