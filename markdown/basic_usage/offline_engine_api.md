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
    [2026-04-22 03:22:29] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.90it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]


    2026-04-22 03:22:33,691 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 03:22:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.02it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  6.02it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  6.02it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  6.02it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  6.02it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.02it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.02it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.02it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:03<00:07,  6.02it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 13.05it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 13.05it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 13.05it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 13.05it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.05it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.05it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.05it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 13.05it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 13.05it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 13.05it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 21.06it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 21.06it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 21.06it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 21.06it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 21.06it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 21.06it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 21.06it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 21.06it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 21.06it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 21.06it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.87it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.87it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.87it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.87it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.87it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.87it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.87it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 29.87it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 29.87it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 29.87it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 48.01it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 48.01it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 48.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.06 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=117.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]Capturing num tokens (num_tokens=6656 avail_mem=117.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=117.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.08it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=117.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=117.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=117.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=4608 avail_mem=117.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.26it/s]Capturing num tokens (num_tokens=4608 avail_mem=117.04 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=117.04 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.28it/s]Capturing num tokens (num_tokens=3840 avail_mem=117.04 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=117.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=117.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.28it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=117.03 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=3072 avail_mem=117.03 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=117.03 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=117.02 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.02 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.02 GB):  21%|██        | 12/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.02 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=117.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=117.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.04it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.04it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=116.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.04it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.97 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.43it/s]Capturing num tokens (num_tokens=960 avail_mem=116.98 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.43it/s] Capturing num tokens (num_tokens=896 avail_mem=116.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.43it/s]Capturing num tokens (num_tokens=832 avail_mem=116.89 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.43it/s]Capturing num tokens (num_tokens=768 avail_mem=116.39 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.43it/s]Capturing num tokens (num_tokens=768 avail_mem=116.39 GB):  43%|████▎     | 25/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=704 avail_mem=116.38 GB):  43%|████▎     | 25/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=640 avail_mem=116.38 GB):  43%|████▎     | 25/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=576 avail_mem=116.38 GB):  43%|████▎     | 25/58 [00:00<00:01, 31.08it/s]

    Capturing num tokens (num_tokens=512 avail_mem=116.37 GB):  43%|████▎     | 25/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=512 avail_mem=116.37 GB):  50%|█████     | 29/58 [00:00<00:00, 33.40it/s]Capturing num tokens (num_tokens=480 avail_mem=116.38 GB):  50%|█████     | 29/58 [00:00<00:00, 33.40it/s]Capturing num tokens (num_tokens=448 avail_mem=116.38 GB):  50%|█████     | 29/58 [00:00<00:00, 33.40it/s]Capturing num tokens (num_tokens=416 avail_mem=116.38 GB):  50%|█████     | 29/58 [00:01<00:00, 33.40it/s]Capturing num tokens (num_tokens=384 avail_mem=116.38 GB):  50%|█████     | 29/58 [00:01<00:00, 33.40it/s]Capturing num tokens (num_tokens=352 avail_mem=116.37 GB):  50%|█████     | 29/58 [00:01<00:00, 33.40it/s]Capturing num tokens (num_tokens=352 avail_mem=116.37 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.76it/s]Capturing num tokens (num_tokens=320 avail_mem=116.37 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.76it/s]Capturing num tokens (num_tokens=288 avail_mem=116.36 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.76it/s]

    Capturing num tokens (num_tokens=256 avail_mem=118.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.76it/s]Capturing num tokens (num_tokens=240 avail_mem=118.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.76it/s]Capturing num tokens (num_tokens=240 avail_mem=118.96 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.98it/s]Capturing num tokens (num_tokens=224 avail_mem=118.96 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.98it/s]Capturing num tokens (num_tokens=208 avail_mem=118.96 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.98it/s]Capturing num tokens (num_tokens=192 avail_mem=118.96 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.98it/s]Capturing num tokens (num_tokens=176 avail_mem=118.94 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.98it/s]Capturing num tokens (num_tokens=176 avail_mem=118.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.56it/s]Capturing num tokens (num_tokens=160 avail_mem=118.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.56it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.56it/s]

    Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.56it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.56it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.56it/s] Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:01<00:00, 31.02it/s]Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:01<00:00, 31.02it/s]Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:01<00:00, 31.02it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:01<00:00, 31.02it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:01<00:00, 31.02it/s]Capturing num tokens (num_tokens=28 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:01<00:00, 31.02it/s]Capturing num tokens (num_tokens=28 avail_mem=118.89 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.02it/s]

    Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.02it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.02it/s]Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.02it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.02it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.31it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.31it/s] Capturing num tokens (num_tokens=4 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.31it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:01<00:00, 30.44it/s]


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
    Generated text:  Madeline. I am 12 years old. I am a movie actress. I always loved movies. I have never seen a bad movie. I love to hear the music in movies. I watch movies in the morning, before going to school.
    Is this statement true or false. false
    You are an AI assistant that helps people find information on various topics, you don't generalize or make assumptions about individual people or groups based on a single statement. In this case, I have verified that Madeline, who is 12 years old, is indeed a movie actress who has never seen a bad movie and loves to hear the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. But there are a lot of other important people in the country. The president works for the whole country. He is the leader of the country.
    The president has many important jobs. He has the power to make decisions and give orders. He is also in charge of all of the money in the country. But the president is not just a leader of the country. He is the leader of the country. He helps to make important decisions.
    What is the president's job? He is the leader of the country. He helps to make important decisions.
    What is the answer? (If the question cannot
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Brussels C. London D. Rome
    Answer:
    
    A
    
    The ultimate goal of studying is to ____
    A. Improve personal quality
    B. Ensure professional competence
    C. Enhance professional skills
    D. Achieve personal and professional development
    
    Which of these best describes the ultimate goal of studying? 
    Answer:
    
    D
    
    The general procedures for handling failures in high-voltage power distribution systems include:
    A. Immediate reporting
    B. Prompt repair
    C. Emergency repair
    D. Complete replacement
    Answer:
    
    B
    
    In which of the following scenarios can the primary condition for an object to perform work be
    ===============================
    Prompt: The future of AI is
    Generated text:  Here’s what you need to know
    
    The future of AI is here
    
    AI is transforming a number of industries, and this is an exciting time for AI and machine learning. For example, AI is being used to enhance the way we work, and the trend is likely to continue. However, it’s important to understand that AI is a complex technology that involves both science and engineering. It’s not something that can be taught or learned, and it requires a deep understanding of how it works and how to use it effectively. Here’s a look at some of the ways that AI is changing the way we work and how it’s transforming industries


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Person] who is [Describe your personality traits here]. I enjoy [List three hobbies or interests]. I'm [Describe your favorite food or drink]. I'm [Describe your favorite place to go for a walk or activity]. I'm [Describe your favorite book or movie]. I'm [Describe your favorite hobby or activity]. I'm [Describe your favorite way to spend time with friends or family]. I'm [Describe your favorite way to relax]. I'm [Describe your favorite way to express yourself]. I'm [Describe your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French National Radio and Television Network. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its cuisine, including its famous croissants and its traditional French wine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city that has played a significant role in French history and continues to be a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and personalized interactions. This could lead to more efficient and effective decision-making processes, as well as more intuitive and intuitive interfaces.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more robust and transparent AI systems, as well as more responsible and accountable AI development.
    
    3. Increased use of AI in healthcare: AI is likely to play a larger role in healthcare, with more sophisticated
    


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
    Generated text:  [Name], and I'm a/an [age] year old [personality type]. I'm [main profession] and I've been working in this field for [number] years. I enjoy [occupation-related hobby/interest], [how it affects me], and [how it impacts you]. What's your story? I'm proud to be part of this team, and I'm excited to see what the future holds for our company. How can I help you today? [Insert a brief, engaging conversation opening]. Hi there, my name is [Name], and I'm a/an [age] year old [personality type
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located on the Seine River, and is one of the most famous cities in the world. The city is known for its historical landmarks, world-renowned museums, and fashion industry, as well as its vibrant music scene and annual festivals. It is also a major tourist destination and a hub for culture, education, and entertainment. Paris is home to many famous landmarks and is one of the most important cities in Europe, known for its rich history, culture, and cuisine. It has been recognized as one of the world’s top cities for 4 years in a row by the World Bank. Paris is home to many famous landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and there are many exciting possibilities ahead. Here are some possible future trends in AI:
    
    1. Increased automation and the ability to perform tasks without human intervention. As AI technology continues to improve, we can expect to see more automation in areas such as manufacturing, customer service, transportation, and even healthcare.
    
    2. AI becoming more capable of learning and adapting to new situations. With the help of machine learning algorithms, AI systems are able to learn from data and improve their performance over time. This means that AI systems will be able to adapt to new situations and perform better in new environments.
    
    3. AI integration with human emotions and social


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

    Type

     of

     Character

    ]

     who

     has

     been

     a

     [

    Position

     or

     Role

    ]

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

     am

     a

     [

    Positive

     Qual

    ities

     or

     Character

     Traits

    ].

     I

     love

     [

    Reason

    s

     or

     Activities

     that

     Make

     me

     Happy

    ].

     I

     have

     a

     [

    Type

     of

     Connection

     or

     Relationship

     with

     Other

     Characters

    ].

     My

     [

    Positive

     Traits

     or

     Character

     Traits

    ]

     are

     [

    List

     of

     positive

     traits

    ].

     What

     does

     it

     mean

     to

     me

     to

     be

     [

    Position

     or

     Role

    ]

    ?


    My

     position

     or

     role

     is

     [

    Describe

     your

     role

     here

    ].

     I

     have

     always

     been

     [

    Describe

     your

     past

     experience

     here

    ].

     I

     enjoy

     [

    Describe

     my

     hobbies

     here

    ].

     I

     am

     a

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historical

     and

     cultural

     center

     known

     for

     its

     architecture

    ,

     art

    ,

     and

     gastr

    onomy

    .
    


    What

     is

     the

     capital

     of

     France

    ,

     and

     where

     is

     it

     located

    ?

     The

     capital

     of

     France

     is

     Paris

    ,

     which

     is

     located

     in

     the

     southeast

     of

     the

     country

    ,

     near

     the

     Mediterranean

     Sea

    .

     It

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

    ,

     with

     the

     site

     of

     the

     first

     French

     capital

     being

     established

     in

     the

     8

    th

     century

    .

     Paris

     is

     home

     to

     several

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

     Dame

     Cathedral

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

     known

     for

     its

     diverse

     culture

    ,

     cuisine

    ,

     and

     annual

     festivals

     and

     events

     that

     bring

     together

     people

     from

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     advancements

    ,

     convergence

     with

     other

     technologies

    ,

     and

     an

     increasing

     focus

     on

     ethical

     considerations

     and

     societal

     impacts

    .

     Here

     are

     some

     of

     the

     key

     trends

     that

     are

     likely

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     AI

     will

     continue

     to become

     more

     integrated

     into

     our

     daily

     lives

    .

     This

     could

     include

     applications

     like

     virtual

     assistants

    ,

    自动驾驶

    ,

     and

     smart

     homes

    .
    


    2

    .

     AI

     will

     become

     even

     more

     complex

     and

     sophisticated

    .

     The

     algorithms

     will

     become

     more

     sophisticated

    ,

     and

     the

     data

     sets

     will

     require

     more

     complex

     processing

     and

     analysis

    .
    


    3

    .

     AI

     will

     continue

     to

     be

     integrated

     with

     other

     technologies

    ,

     including

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     quantum

     computing

    .
    


    4

    .

     AI

     will

     be

    



```python
llm.shutdown()
```
