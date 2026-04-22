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
    [2026-04-22 23:42:53] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.37it/s]


    2026-04-22 23:42:57,951 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 23:42:57] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.72it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.52it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.52it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.19it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.19it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.19it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.19it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.19it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.19it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.19it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.19it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.19it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.19it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 47.92it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 47.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=136.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=136.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=136.72 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=7168 avail_mem=136.71 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=6656 avail_mem=136.71 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=136.71 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=136.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=136.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=136.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=136.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=136.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=136.70 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3840 avail_mem=136.70 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3584 avail_mem=136.70 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=136.69 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=136.69 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=136.69 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.31it/s]Capturing num tokens (num_tokens=2816 avail_mem=136.69 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.31it/s]Capturing num tokens (num_tokens=2560 avail_mem=136.68 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=136.68 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=136.68 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=136.67 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=136.67 GB):  31%|███       | 18/58 [00:00<00:01, 35.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=136.67 GB):  31%|███       | 18/58 [00:00<00:01, 35.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=136.67 GB):  31%|███       | 18/58 [00:00<00:01, 35.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=136.65 GB):  31%|███       | 18/58 [00:00<00:01, 35.51it/s]

    Capturing num tokens (num_tokens=960 avail_mem=136.66 GB):  31%|███       | 18/58 [00:00<00:01, 35.51it/s] Capturing num tokens (num_tokens=896 avail_mem=136.66 GB):  31%|███       | 18/58 [00:00<00:01, 35.51it/s]Capturing num tokens (num_tokens=896 avail_mem=136.66 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=832 avail_mem=136.65 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=768 avail_mem=136.65 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=704 avail_mem=136.65 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=640 avail_mem=136.64 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=576 avail_mem=136.64 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=576 avail_mem=136.64 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.80it/s]Capturing num tokens (num_tokens=512 avail_mem=136.63 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.80it/s]Capturing num tokens (num_tokens=480 avail_mem=136.65 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.80it/s]

    Capturing num tokens (num_tokens=448 avail_mem=136.65 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.80it/s]Capturing num tokens (num_tokens=416 avail_mem=136.64 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.80it/s]Capturing num tokens (num_tokens=384 avail_mem=136.64 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.80it/s]Capturing num tokens (num_tokens=384 avail_mem=136.64 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=352 avail_mem=136.64 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=320 avail_mem=136.63 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=288 avail_mem=136.63 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=256 avail_mem=136.63 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=240 avail_mem=136.62 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=240 avail_mem=136.62 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=224 avail_mem=136.62 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.78it/s]

    Capturing num tokens (num_tokens=208 avail_mem=136.62 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=192 avail_mem=136.62 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=176 avail_mem=136.61 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=160 avail_mem=136.61 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=160 avail_mem=136.61 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=144 avail_mem=136.61 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=128 avail_mem=136.60 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=112 avail_mem=136.60 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=96 avail_mem=136.60 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.43it/s] Capturing num tokens (num_tokens=80 avail_mem=136.59 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.43it/s]

    Capturing num tokens (num_tokens=80 avail_mem=136.59 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=64 avail_mem=136.59 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=48 avail_mem=136.59 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=32 avail_mem=136.58 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=28 avail_mem=136.58 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=24 avail_mem=136.58 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=24 avail_mem=136.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=20 avail_mem=136.57 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=16 avail_mem=136.57 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=12 avail_mem=136.57 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=8 avail_mem=136.56 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.80it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=136.56 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=4 avail_mem=136.56 GB): 100%|██████████| 58/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=4 avail_mem=136.56 GB): 100%|██████████| 58/58 [00:01<00:00, 38.82it/s]


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
    Generated text:  Elena and I am a doctor. I’m from Russia and I'm very kind to you. I have my own opinion about my country and its people. I'm a woman and I'm very strong and I care about people. I'm very proud of me. I like to play in the snow and fishing in the rivers. I live in a small town in the forest. I like to have fun. I like to play with my children and have some time for myself. I do not live alone. I have a dog. I also have a cat. I like to give gifts. I have one family. I have one daughter
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. He estimates that the total population of the country is 250 million. He is considering three types of bases: conventional bases, nuclear bases, and cyber bases. Each conventional base requires 50,000 square meters of land, each nuclear base requires 100,000 square meters of land, and each cyber base requires 200,000 square meters of land. He also considers that each conventional base uses 100 gallons of fuel per month, each nuclear base uses 300 gallons of fuel per month, and
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Marseille
    C. Lyon
    D. Toulouse
    
    The capital of France is Paris. Therefore, the correct answer is:
    A. Paris
    B. Marseille
    C. Lyon
    D. Toulouse
    A. Paris is the capital of France, while Marseille, Lyon, and Toulouse are cities in France. Therefore, the correct answer is:
    A. Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but it will change the way we live, work and communicate. With AI, we are able to automate many tasks that would otherwise take humans days or weeks to complete. The more AI is integrated into our lives, the more we will be exposed to potential security risks. If not properly regulated, AI could potentially harm the planet.
    For the past few years, the world has seen a surge of interest in the use of AI to help solve the world’s biggest problems. In the last few years, governments and businesses have invested heavily in AI and are looking to make use of it in their decision-making processes. As AI becomes more


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and the Middle Ages. Paris is home to many famous museums, including the Louvre and the Musée d'Orsay, as well as the Notre-Dame Cathedral. The city is also known for its fashion industry, with many famous fashion houses and boutiques located in the city center. Paris is a vibrant and diverse city with a rich cultural heritage that continues to attract visitors from around the world
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there is a growing emphasis on developing AI that is designed to be ethical and responsible. This could involve developing AI that is transparent, accountable, and accountable to humans.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As the technology continues to evolve, we can expect to see even more widespread use of AI in healthcare, with more
    


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
    Generated text:  [Your Name] and I am a [Your Profession] who specializes in [Your Area of Expertise]. I have been working at [Your Company Name] for [Your Duration] years. I have a passion for [Your Passion/Interest] and am always looking for ways to [Your Career Goal or Motivation]. I have a strong work ethic and am always seeking opportunities to grow and learn. I am a [Your Personality Type] and I am always ready to take on new challenges and work hard towards achieving my goals. Overall, I am a [Your Overall Character Trait] and I am committed to helping others succeed. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    You are an AI assistant that helps you understand the phrases in French. Can you tell me what the French word for "star" means?
    
    Sure! The French word for "star" is "lune". In this case, "lune" is a compound word meaning "moon" or "lunar". So, Paris, the capital city of France, is often referred to as the "lunar capital" or simply "lunar city". To put it simply, Paris is indeed the "lunar capital" of France! 🌹✨
    
    The French word for "star" is "lune", and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and diverse, with potential applications in many different fields. Some possible trends include:
    
    1. Increased autonomy and decision-making: AI is becoming more capable of making decisions on its own without human intervention, which could lead to more autonomous systems and intelligent agents.
    
    2. Enhanced privacy and data protection: As AI systems become more advanced, there will be increased concerns about privacy and data protection. Developers will need to develop new strategies to ensure that AI systems are secure and protect sensitive data.
    
    3. More reliance on AI in healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and this trend is expected to continue. AI


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

     Profession

    ]

     [

    Here

     is

     the

     profession

     for

     any

     other

     professions

     in

     your

     character

    ’s

     world

    ].

     I

    'm

     [

    Your

     Age

    ]

     years

     old

    ,

     and

     I

    've

     always

     been

     an

     [

    Your

     Character

     Trait

    ],

     [

    Here

     is

     an

     example

     of

     how

     you

     might

     use

     this

     trait

     in

     a

     fictional

     scenario

    ].

     I

     love

     [

    Your

     Hobby

     or

     Passion

    ],

     [

    Here

     is

     an

     example

     of

     how

     you

     might

     describe

     your

     hobby

     or

     passion

     in

     a

     fictional

     scenario

    ].

     And

     I

    'm

     always

     [

    Your

     Character

     Quality

    ,

     [

    Here

     is

     an

     example

     of

     how

     you

     might

     describe

     your

     quality

     in

     a

     fictional

     scenario

    ]].

     I

     have

     a

     passion

     for

     [

    Your

     Career

     Goal

    ],

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    The

     City

     of

     Light

    ".

     It

     is

     a

     cosm

    opolitan

     city

     with

     a

     rich

     history

     and

     a

     beautiful

     medieval

     city

     center

    .

     Paris

     is

     home

     to

     the

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

     as

     well

     as

     a

     vibrant

     arts

     and

     culture

     scene

    .

     The

     city

     is

     also

     known

     for

     its

     food

    ,

     music

    ,

     and

     fashion

    ,

     with

     many

     famous

     chefs

     and

     musicians

    .

     In

     recent

     years

    ,

     Paris

     has

     experienced

     a

     significant

     population

     growth

     due

     to

     its

     status

     as

     a

     global

     capital

    .

     As

     of

     

    2

    0

    2

    1

    ,

     Paris

     had

     a

     population

     of

     approximately

     

    2

    .

    1

     million

     people

    .

     It

     is

     home

     to

     many

     museums

    ,

     art

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     four

     key

     trends

    :
    


    1

    .

     Increased

     availability

     and

     affordability

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     the

     costs

     of

     implementing

     AI

     will

     decrease

    ,

     making

     it

     more

     accessible

     to

     organizations

     of

     all

     sizes

    .

     This

     will

     allow

     for

     a

     wider

     range

     of

     AI

     applications

     to

     be

     developed

     and

     deployed

    .
    


    2

    .

     Greater

     integration

     with

     other

     technologies

    :

     AI

     will

     likely

     become

     increasingly

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

    ,

     IoT

     devices

    ,

     and

     machine

     learning

     algorithms

    .

     This

     will

     allow

     for

     more

     complex

     and

     sophisticated

     AI

     systems

     to

     be

     built

    ,

     leading

     to

     breakthrough

    s

     in

     areas

     such

     as

     autonomous

     vehicles

     and

     healthcare

    .
    


    3

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     integrated

     into

    



```python
llm.shutdown()
```
