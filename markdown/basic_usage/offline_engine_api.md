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
    [2026-04-22 14:36:04] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]


    2026-04-22 14:36:09,511 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 14:36:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.72it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 13.03it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:02, 13.03it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 29.68it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 29.68it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 29.68it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 29.68it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 29.68it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 29.68it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 29.68it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 29.68it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 29.68it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:03<00:00, 29.68it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 38.91it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 48.14it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 48.14it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 48.14it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 48.14it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 48.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.39 GB):   3%|▎         | 2/58 [00:00<00:03, 16.92it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.38 GB):   3%|▎         | 2/58 [00:00<00:03, 16.92it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.38 GB):   3%|▎         | 2/58 [00:00<00:03, 16.92it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.38 GB):   3%|▎         | 2/58 [00:00<00:03, 16.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.37 GB):   9%|▊         | 5/58 [00:00<00:02, 20.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.37 GB):   9%|▊         | 5/58 [00:00<00:02, 20.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.65it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.65it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.65it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=116.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.33 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.33 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.32 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.32 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.31 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.31 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.31 GB):  31%|███       | 18/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.31 GB):  31%|███       | 18/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.30 GB):  31%|███       | 18/58 [00:00<00:01, 34.67it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=116.28 GB):  31%|███       | 18/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=960 avail_mem=116.30 GB):  31%|███       | 18/58 [00:00<00:01, 34.67it/s] Capturing num tokens (num_tokens=896 avail_mem=116.29 GB):  31%|███       | 18/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=896 avail_mem=116.29 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.04it/s]Capturing num tokens (num_tokens=832 avail_mem=116.29 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.04it/s]Capturing num tokens (num_tokens=768 avail_mem=116.29 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.04it/s]Capturing num tokens (num_tokens=704 avail_mem=116.29 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.04it/s]Capturing num tokens (num_tokens=640 avail_mem=116.28 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.04it/s]Capturing num tokens (num_tokens=576 avail_mem=116.28 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.04it/s]Capturing num tokens (num_tokens=576 avail_mem=116.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.63it/s]Capturing num tokens (num_tokens=512 avail_mem=116.27 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.63it/s]

    Capturing num tokens (num_tokens=480 avail_mem=116.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.63it/s]Capturing num tokens (num_tokens=448 avail_mem=116.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.63it/s]Capturing num tokens (num_tokens=416 avail_mem=116.28 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.63it/s]Capturing num tokens (num_tokens=416 avail_mem=116.28 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.14it/s]Capturing num tokens (num_tokens=384 avail_mem=116.28 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.14it/s]Capturing num tokens (num_tokens=352 avail_mem=116.27 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.14it/s]Capturing num tokens (num_tokens=320 avail_mem=116.27 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.14it/s]Capturing num tokens (num_tokens=288 avail_mem=116.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.14it/s]Capturing num tokens (num_tokens=256 avail_mem=116.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.14it/s]Capturing num tokens (num_tokens=256 avail_mem=116.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=240 avail_mem=116.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.63it/s]

    Capturing num tokens (num_tokens=224 avail_mem=116.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=208 avail_mem=116.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=192 avail_mem=116.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=176 avail_mem=116.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=176 avail_mem=116.25 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=160 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=144 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=128 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=112 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=96 avail_mem=116.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.20it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=116.23 GB):  81%|████████  | 47/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=80 avail_mem=116.23 GB):  81%|████████  | 47/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=64 avail_mem=116.22 GB):  81%|████████  | 47/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=48 avail_mem=116.22 GB):  81%|████████  | 47/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=32 avail_mem=116.22 GB):  81%|████████  | 47/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=28 avail_mem=116.21 GB):  81%|████████  | 47/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=28 avail_mem=116.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=24 avail_mem=116.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=20 avail_mem=116.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=16 avail_mem=116.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=12 avail_mem=116.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.38it/s]

    Capturing num tokens (num_tokens=8 avail_mem=116.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.38it/s] Capturing num tokens (num_tokens=8 avail_mem=116.20 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.28it/s]Capturing num tokens (num_tokens=4 avail_mem=116.20 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.28it/s]Capturing num tokens (num_tokens=4 avail_mem=116.20 GB): 100%|██████████| 58/58 [00:01<00:00, 37.35it/s]


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
    Generated text:  Peter.
    I'm a designer and artist, as well as a teacher.
    I believe in the importance of creativity in everyday life.
    I use the creative process to improve my skills and skills to improve my teaching.
    I’ve worked with artists and design professionals for years.
    My experience ranges from working with clients on projects to running workshops with children.
    The projects that I work on are always unique and innovative and will challenge the mind.
    I’ve created many unique artworks and installations, ranging from installations to artworks.
    I’ve also spent time teaching children and working with students, giving them the tools to develop their skills and interests.
    My focus is on improving
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. Which of the following cannot be considered an example of the president of the United States? A. President of the United States of America B. President of the United States of Mexico C. President of the United States of Canada D. President of the United States of Europe
    Answer: D
    
    The following statement is about the characteristics of a perfect market, but it is incorrect. The statement is: "In a perfectly competitive market, all firms are price takers, and there is no collusion among firms."
    A. Correct
    B. Incorrect
    C. Unclear
    D. Cannot be determined
    Answer: B
    
    In
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, but Paris is not its capital.
    The answer to the riddle is Paris. The reason given in the answer is that Paris is the capital of France, but it is not the capital of France. The answer doesn't provide any new information or add to the context of the riddle. It simply reinforces the given information. The riddle is asking what the capital of France is, and the answer is Paris. The capital of France is not Paris. The capital of France is Rome. The capital of France is not Paris. The capital of France is not Paris. The capital of France is Rome. The capital of France is
    ===============================
    Prompt: The future of AI is
    Generated text:  highly dependent on the types of problems that are being solved. In the robotics domain, the types of problems that will be solved by AI are increasing, and the types of AI algorithms that are required to solve these problems will also increase. In this project, we will explore the use of deep learning and reinforcement learning for solving robotics problems.
    
    ### Deep Learning
    
    Deep learning is an important branch of machine learning that enables computers to recognize patterns in complex data. In robotics, deep learning can be used to build intelligent robots that can solve complex problems such as planning, navigation, and learning.
    
    #### 1. Introduction to Deep Learning
    
    Deep learning is


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I enjoy [job title] because [reason for interest]. I'm always looking for ways to [job title] and I'm always eager to learn new things. What's your favorite hobby or activity? I love [hobby or activity], and I enjoy [reason for interest]. I'm always looking for new ways to [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in Europe by population. It is located on the Seine River and is the seat of the French government and the country's cultural, political, and economic center. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its cuisine, fashion, and art, and is home to many museums, theaters, and other cultural institutions. Paris is a vibrant and diverse city with a rich history and a strong sense of identity. Its status
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into various industries, from manufacturing to healthcare to transportation. This automation will likely lead to the development of new types of AI that can perform tasks that are currently performed by humans, such as data analysis, decision-making, and problem-solving.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be an increasing need for privacy and security. This will likely lead to the development of new AI technologies
    


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
    Generated text:  [Name], and I am a [Role] within [Company]. I am [Degree] with [Field of Study] and [Number of Publications]. Here's how I started my journey: [Briefly describe a typical day for me]. My expertise and interests are in [Field of Interest or Career]. I am passionate about [One Interest or Hobby], which drives me to [How the hobby helps me]. My goal is to [Specific Goal or Career Objective]. If you have a question related to [Subject or Topic], I will do my best to answer it. That's all I have for now. How can I assist you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the heart of the country. It is located on the River Seine and the Champ de Mars park, and is home to the Eiffel Tower and many other famous landmarks. The city is known for its rich history and cultural heritage, and is a popular tourist destination for millions of people each year. Its status as the capital makes it the capital of France and one of the most important cities in the world. Paris is also home to the French Parliament, the French Supreme Court, and the French Institute of France. In addition, Paris is the birthplace of many famous artists, writers
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and dependent on several factors, including advances in computing power, the development of new technologies, and the evolving nature of human intelligence and emotions. Here are some possible future trends in AI:
    
    1. Increased Real-World Applications: As AI continues to evolve, we can expect to see more and more applications in real-world settings, such as self-driving cars, personalized medicine, and financial services.
    
    2. Enhanced Privacy and Security: As AI becomes more sophisticated, we can expect to see more emphasis on protecting user privacy and security. This may lead to new technologies and regulations that address issues such as data minimization, consent, and accountability


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

     a

    /an

     [

    insert

     profession

     or

     major

    ]

     student

     from

     [

    insert

     location

    ]

     with

     a

     keen

     interest

     in

     [

    insert

     subject

     of

     interest

    ].

     I

    'm

     an

     [

    insert

     age

    ]

     year

     old

    ,

     and

     I

     enjoy

     [

    insert

     hobbies

     or

     interests

    ]

     outside

     of

     school

    .

     Outside

     of

     school

    ,

     I

    've

     always

     been

     passionate

     about

     [

    insert

     hobby

     or

     interest

    ],

     and

     I

    've

     been

     volunteering

     my

     time

     and

     skills

     for

     [

    insert

     organization

     or

     non

    -profit

    ]

     for

     the

     past

     [

    insert

     number

    ]

     years

    .

     I

     enjoy

     [

    insert

     activities

     that

     I

     enjoy

     doing

     outside

     of

     school

    ,

     such

     as

     [

    insert

     extr

    ac

    ur

    ricular

     activities

     or

     hobbies

     like

     playing

     sports

    ,

     reading

     books

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     a

     city

     in

     eastern

     France

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     culture

    .

     It

    's

     the

     second

     most

     populous

     city

     in

     the

     European

     Union

     and

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     France

    's

     economic

     center

     and

     cultural

     hub

    ,

     Paris

     has

     become

     synonymous

     with

     luxury

     and

     haute

     cuisine

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     which

     draws

     many

     international

     designers

    .

     Despite

     its

     high

     cost

     of

     living

    ,

     Paris

     is

     considered

     one

     of

     the

     world

    's

     most

     desirable

     cities

    .

     The

     city

     is

     also

     home

     to

     many

     museums

     and

     festivals

    .

     Its

     skyline

     is

     marked

     by

     the

     iconic

     E

    iff

    el

     Tower

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     technological

     and

     disruptive

    .

     Here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

     of

     AI

     in

     various

     sectors

    :

     AI

     is

     expected

     to

     become

     more

     pervasive

     in

     various

     industries

    ,

     including

     healthcare

    ,

     transportation

    ,

     finance

    ,

     and

     manufacturing

    .

     This

     integration

     will

     lead

     to

     the

     development

     of

     more

     sophisticated

     AI

     systems

     that

     can

     perform

     tasks

     that

     were

     previously

     performed

     by

     humans

    .
    


    2

    .

     Emer

    gence

     of

     new

     AI

     technologies

    :

     There

     is

     a

     growing

     interest

     in

     developing

     new

     AI

     technologies

     that

     can

     perform

     tasks

     that

     are

     currently

     being

     handled

     by

     humans

    .

     These

     new

     technologies

     could

     include

     robotics

    ,

     natural

     language

     processing

    ,

     and

     machine

     learning

    .
    


    3

    .

    



```python
llm.shutdown()
```
