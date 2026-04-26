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
    [2026-04-26 17:20:31] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.55it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.53it/s]


    2026-04-26 17:20:36,341 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 17:20:36] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.14it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.55it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.55it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.55it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.55it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.55it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.55it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.55it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.55it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.55it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.55it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.48it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.48it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.48it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.48it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.48it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.48it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.48it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.48it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.48it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.48it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 39.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.93 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.90 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.90 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.89 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.89 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.89 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.89 GB):   9%|▊         | 5/58 [00:00<00:02, 20.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.88 GB):   9%|▊         | 5/58 [00:00<00:02, 20.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.83 GB):   9%|▊         | 5/58 [00:00<00:02, 20.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.80 GB):   9%|▊         | 5/58 [00:00<00:02, 20.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.44it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.44it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.79 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.79 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.79 GB):  21%|██        | 12/58 [00:00<00:01, 28.78it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.79 GB):  21%|██        | 12/58 [00:00<00:01, 28.78it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.79 GB):  21%|██        | 12/58 [00:00<00:01, 28.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.78 GB):  21%|██        | 12/58 [00:00<00:01, 28.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.78 GB):  21%|██        | 12/58 [00:00<00:01, 28.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.78 GB):  21%|██        | 12/58 [00:00<00:01, 28.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.78 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.77 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.77 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.12it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=118.77 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.75 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.12it/s]Capturing num tokens (num_tokens=960 avail_mem=118.76 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.12it/s] Capturing num tokens (num_tokens=960 avail_mem=118.76 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=896 avail_mem=118.76 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=832 avail_mem=118.76 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=768 avail_mem=118.75 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=704 avail_mem=118.75 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=640 avail_mem=118.75 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=640 avail_mem=118.75 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.56it/s]Capturing num tokens (num_tokens=576 avail_mem=118.75 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.56it/s]

    Capturing num tokens (num_tokens=512 avail_mem=118.73 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.56it/s]Capturing num tokens (num_tokens=480 avail_mem=118.75 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.56it/s]Capturing num tokens (num_tokens=448 avail_mem=118.74 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.56it/s]Capturing num tokens (num_tokens=416 avail_mem=118.74 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.56it/s]Capturing num tokens (num_tokens=416 avail_mem=118.74 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=384 avail_mem=118.74 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=352 avail_mem=118.73 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=320 avail_mem=118.73 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=288 avail_mem=118.71 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.11it/s]

    Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=224 avail_mem=118.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.85it/s]

    Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.68it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.68it/s] Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.68it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.68it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.68it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=32 avail_mem=118.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.41it/s]

    Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=16 avail_mem=118.63 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.95it/s] Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:01<00:00, 36.86it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:01<00:00, 35.68it/s]


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
    Generated text:  Janelle and I am the author of "The Treasure of Honey". In this book, I introduced you to the art of making the sweetest flavors of honey, and how it has become a cultural treasure in our diets. This book will help you learn how to make your own honey at home and how to use it in a variety of recipes.
    Here's a quick, easy recipe for the perfect glass jam jar. Just add some sugar and honey to the bottom of your glass jar, and then add a few tablespoons of lemon juice and a pinch of sea salt. Screw on the lid, and you're all set!
    When making
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ________ person. [ ]
    A. excellent
    B. ambitious
    C. prosperous
    D. capable
    Answer:
    D
    
    The new medical drugs are more cost-effective than the old ones. ____
    Answer:
    D
    
    The weather forecast says that the day _______ tomorrow is going to be rainy. [ ]
    A. that
    B. what
    C. which
    D. as
    Answer:
    C
    
    The company has decided to _____ its own production line in order to improve efficiency. [ ]
    A. put
    B. put off
    C. put down
    D. put out
    Answer:
    A
    
    The police arrested
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris
    B. Strasbourg
    C. Nantes
    D. La Défense
    The capital of France is Paris. 
    
    Therefore, the correct answer is:
    
    A. Paris. 
    
    (Note: I will provide a brief explanation for each option to ensure accuracy):
    
    B. Strasbourg: The capital of France is not Strasbourg. The capital is typically in the heart of the country and is often the largest city, known for its vibrant culture, art, and historical significance.
    
    C. Nantes: The capital of France is not Nantes. The capital is typically in the heart of the country and is often the
    ===============================
    Prompt: The future of AI is
    Generated text:  changing the world. Artificial intelligence has been in the news lately due to several notable news articles, from the first time the term was coined, to recent news articles on the topic. It’s hard to say how long the topic has existed, as there are many definitions of AI and how it is used in our society. However, there is a clear explanation for the term: it’s the ability of a computer to perform tasks that usually require a human to perform.
    It’s a job for a computer to be able to perform tasks that people usually do, and it’s a part of the job for a computer to be able to learn from


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Vehicle] with [Number of Wheels] wheels. I have [Number of Feet] feet and [Number of Hands] hands. I am [Gender] and [Race]. I am [Occupation] with [Number of Children]. I am [Gender] and [Race]. I am [Occupation] with [Number of Children]. I am [Gender] and [Race]. I am [Occupation] with [Number of Children]. I am [Gender] and [Race]. I am [Occupation] with [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is also home to many notable French writers, artists, and musicians. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is known for its fashion industry, with many famous fashion houses and boutiques. Overall, Paris is a vibrant and exciting city that is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Enhanced privacy and security: As AI systems become more integrated with human intelligence, there will be an increased need for privacy and security measures to protect sensitive data. This could lead to the development of new technologies and protocols that
    


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
    Generated text:  [Name], and I'm a [Title] at [Company]. I'm excited to meet you and learn about your career goals. Let's connect and discuss how we can work together to achieve [career goal]. Are you ready to start your journey with me? [Name]: Welcome, [Name]. I'm [Your Name] and I'm excited to chat with you. What's your professional background, and how can I help you in your career goals? [Name]: Well, [Name], I have a degree in [Field of Study] and over [Number] years of experience in [Field of Study]. I have a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the European Union and the second most populous city in the world. Paris is a historic city with a rich history dating back to the Middle Ages. It is famous for its historical landmarks, art museums, and iconic architecture such as the Eiffel Tower and the Louvre. The city is known for its fashion, gastronomy, and food culture, as well as for its romantic culture and romantic architecture. Paris is also a major hub for international trade and diplomacy, with many world-renowned museums and monuments. Paris has a rich cultural scene and hosts numerous cultural events and festivals throughout the year, including
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, with significant opportunities and challenges ahead. Here are some possible trends in AI:
    
    1. Increased Efficiency and Automation: AI is expected to increase efficiency in various industries, automating tasks that are currently performed manually, saving time and reducing costs.
    
    2. Personalization and Tailored Experiences: AI will enable more personalized experiences for users, with algorithms that learn from data to provide tailored recommendations and services.
    
    3. Integration of AI and Robotics: As AI becomes more advanced, it will integrate more with robotics, creating more robotic systems that are more intelligent and capable.
    
    4. Ethical and Responsible AI: As more AI is being developed,


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

    Your

     Name

    ],

     and

     I

    'm

     a

     [

    Your

     Occupation

    ]

     with

     a

     passion

     for

     [

    Your

     Hobby

    ].

     I

    'm

     constantly

     learning

     and

     growing

    ,

     constantly

     striving

     to

     improve

     my

     skills

     and

     knowledge

    .

     I

     believe

     in

     the

     power

     of

     [

    Your

     Hobby

    ]

     to

     inspire

     and

     motivate

     me

    ,

     and

     I

     always

     strive

     to

     make

     the

     world

     a

     better

     place

     by

     using

     my

     skills

     and

     knowledge

    .

     So

    ,

     if

     you

     ever

     have

     a

     question

     or

     want

     to

     discuss

     a

     hobby

    ,

     feel

     free

     to

     reach

     out

     to

     me

    !

     [

    Your

     Name

    ]

     [

    Your

     Profession

    ]

     I

    'm

     always

     eager

     to

     share

     my

     knowledge

     and

     experiences

     with

     anyone

     who

     would

     like

     to

     learn

    .

     I

     hope

     to

     inspire

     and

     motivate

     you

     to

     do

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     historic

     city

     with

     a

     rich

     history

     and

     cultural

     significance

    .

     The

     city

     is

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

     and

     the

     E

    iff

    el

     Tower

    ,

     as

     well

     as

     its

     bustling

     streets

     and

     lively

     nightlife

    .

     Paris

     is

     also

     famous

     for

     its

     arts

     and

     entertainment

     scenes

    ,

     including

     the

     Op

    éra

     Garn

    ier

     and

     the

     Mou

    lin

     Rouge

    .

     The

     city

     has

     a

     strong

     emphasis

     on

     French

     culture

     and

     is

     home

     to

     numerous

     French

     cuisine

     dishes

    .

     It

     is

     a

     popular

     tourist

     destination

     for

     visitors

     from

     around

     the

     world

    .

     France

    ’s

     capital

     city

    ,

     Paris

    ,

     is

     known

     for

     its

     iconic

     landmarks

    ,

     vibrant

     street

     life

    ,

     and

     delicious

     cuisine

    ,

     making

     it

     a

     must

    -

    visit

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     diverse

     and

     rapidly

     evolving

    ,

     and

     there

     are

     many

     potential

     trends

     and

     areas

     of

     focus

     that

     we

     can

     expect

     to

     see

     in

     the

     years

     to

     come

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

     use

     of

     AI

     for

     autonomous

     and

     automated

     decision

    -making

    :

     With

     the

     increasing

     demand

     for

     more

     efficient

     and

     accurate

     decision

    -making

     processes

    ,

     we

     can

     expect

     to

     see

     more

     AI

     systems

     being

     used

     for

     autonomous

     and

     automated

     decision

    -making

     in

     areas

     such

     as

     transportation

    ,

     healthcare

    ,

     and

     criminal

     justice

    .
    


    2

    .

     Enhanced

     human

     interaction

     and

     empathy

    :

     As

     AI

     becomes

     more

     advanced

    ,

     we

     may

     see

     an

     increase

     in

     the

     amount

     of

     AI

     that

     is

     used

     to

     assist

     with

     human

     interactions

     and

     emotions

    ,

     such

     as

     through

    



```python
llm.shutdown()
```
