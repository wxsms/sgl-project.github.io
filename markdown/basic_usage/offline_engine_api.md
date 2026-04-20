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
    [2026-04-20 15:36:50] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.26it/s]


    2026-04-20 15:36:55,214 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-20 15:36:55] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:30,  1.77it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:11,  4.19it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:11,  4.19it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:11,  4.19it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:11,  4.19it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:03<00:11,  4.19it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  7.15it/s]

    Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.52it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.52it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.52it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.52it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.52it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.52it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.52it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.52it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 19.55it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 19.55it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 19.55it/s]

    Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 19.55it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 19.55it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 19.55it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 19.55it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 19.55it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:00, 27.15it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:00, 27.15it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:00, 27.15it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:00, 27.15it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:00, 27.15it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:00, 27.15it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:00, 27.15it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:00, 27.15it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 34.36it/s]

    Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 41.18it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 41.18it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 41.18it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 41.18it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 41.18it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 41.18it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 41.18it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 41.18it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 47.07it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 47.07it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 47.07it/s]

    Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 47.07it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 47.07it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 47.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.08it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   2%|▏         | 1/58 [00:00<00:05,  9.84it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:05,  9.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:05,  9.84it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.56it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.56it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   9%|▊         | 5/58 [00:00<00:04, 11.87it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:04, 11.87it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:04, 11.87it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.76 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.69it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.20it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.58it/s]Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.58it/s] Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.58it/s]Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.58it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.58it/s]

    Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.46it/s]Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.46it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.46it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.46it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.46it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  50%|█████     | 29/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 29.72it/s]

    Capturing num tokens (num_tokens=384 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=384 avail_mem=118.70 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.03it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.03it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.03it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.03it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.03it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=224 avail_mem=118.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.57it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.57it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  71%|███████   | 41/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  71%|███████   | 41/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  71%|███████   | 41/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  71%|███████   | 41/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  71%|███████   | 41/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  71%|███████   | 41/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.59it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.59it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.59it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.59it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.59it/s]Capturing num tokens (num_tokens=32 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.59it/s]Capturing num tokens (num_tokens=32 avail_mem=118.64 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.67it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.67it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.67it/s]Capturing num tokens (num_tokens=20 avail_mem=118.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.67it/s]Capturing num tokens (num_tokens=16 avail_mem=118.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 36.67it/s]

    Capturing num tokens (num_tokens=16 avail_mem=118.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  95%|█████████▍| 55/58 [00:02<00:00, 36.44it/s] Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  95%|█████████▍| 55/58 [00:02<00:00, 36.44it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 27.96it/s]


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
    Generated text:  Mark and I am a young, ambitious software engineer with a keen interest in technology and innovation. I am currently pursuing my PhD in Computer Science at the University of Toronto, Canada.
    I have always been fascinated by the intersection of technology and society, and I am eager to explore this area further. I am particularly interested in how technology can be used to promote social justice and equity.
    My background in software engineering has allowed me to work on a wide range of projects that require problem-solving, technical understanding, and creativity. I have a deep appreciation for the power of data analysis and the ability to model complex systems.
    I am excited to contribute to
    ===============================
    Prompt: The president of the United States is
    Generated text:  a four-term elected office that is filled by a direct election. The incumbent president is Donald Trump, who was elected in 2016. During the election, candidates were first ranked by the percentage of the popular vote, and then by the number of electoral votes. Trump was ranked 1st by popular vote and 4th by electoral vote. 
    What is the result of the election between Trump and Biden?
    To determine the result of the election between Donald Trump and Joe Biden, we need to follow the steps of the election process:
    
    1. **Determine the first place ranking**:
       - Trump was ranked 1
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Berlin
    C. Washington
    D. Moscow
    The correct answer is A. Paris.
    
    The capital of France is Paris, which is known for its beautiful architecture, vibrant culture, and many museums and historical landmarks. The other options, Berlin, Washington, and Moscow, are not capitals of countries.
    ===============================
    Prompt: The future of AI is
    Generated text:  secure
    
    I am a big believer in a return to the principles of the Internet of Things, not the blockchain. In that light, the idea of a secure AI system seems to fit perfectly.
    
    If you haven’t heard, the world has been getting more and more concerned about AI safety. Imagine if you were browsing the web and you saw that some AI has a significant risk of being hacked. You’d know right away that that AI is risky. It’s a clear sign that the system is not as secure as it could be. And the more your web browser shows up the more and more alarming the message becomes.
    
    We all know what


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, interesting fact about yourself]. And what can you tell me about your company? I'm excited to learn more about you and your company. What can you tell me about your company? I'm excited to learn more about you and your company. What can you tell me about your company? I'm excited to learn more about you and your company. What can you tell me about your company? I'm excited to learn more
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also a popular tourist destination known for its rich history, art, and cuisine. Paris is a vibrant and diverse city with a rich cultural heritage that continues to attract visitors from around the world. It is a city that has played a significant role in shaping French identity and culture for centuries. The city is also home to many notable museums, including the Musée d'Orsay and the Musée Rodin. Paris is a city that is a must-visit for anyone interested in French culture and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more robust AI systems that are designed to be transparent, accountable, and responsible.
    
    3
    


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
    Generated text:  [Name] and I'm a [Main Character's occupation or special ability]. I've always been fascinated by the world of [Subtle World], and I'm always eager to learn about it. My [Subtle World] expertise is [Describe how you've gained knowledge or experience in the world you study]. I believe in the power of empathy and compassion, and I strive to use my abilities to help those in need. I'm always looking for new challenges to tackle and a place to learn from new perspectives. Thank you for having me. [Name] feels confident, grounded, and ready to embrace whatever adventures life may throw at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris, also known as “La France,” is the capital city of the country, located in the north-central region of France, on the Île de la Cité, on the river Seine, and in the city center of the city. Paris is the third most populous city in the world, after Tokyo and New York City, and is the largest city in the European Union. It is also the most-visited city in the world, and the only city in the world to be both a UNESCO World Heritage site and a city with a population over one million. It is also the birthplace of many of the world
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a range of emerging trends and technologies that are expected to continue to drive innovation and impact various industries. Here are some possible future trends in artificial intelligence:
    
    1. Increased AI-powered automation: As AI technology continues to improve, automation will become more prevalent in many industries, from manufacturing to healthcare to customer service. This will lead to increased efficiency, cost savings, and productivity gains for businesses.
    
    2. AI-powered personal assistants: As AI becomes more advanced, personal assistants will become more integrated into our daily lives, providing us with a range of useful services such as voice recognition, speech recognition, and image recognition.
    
    3.


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

    'm

     a

     [

    brief

     but

     compelling

     description

     of

     your

     character

    ].

     Throughout

     our

     conversation

    ,

     we

    'll

     be

     exploring

     the

     world

     of

     [

    the

     fictional

     place

    ,

     product

    ,

     or

     service

    ]

     and

     discussing

     the

     interesting

     aspects

     of

     [

    their

     role

     or

     profession

    ].

     Remember

    ,

     this

     is

     just

     a

     brief

     introduction

    ,

     and

     I

     can

    't

     reveal

     too

     much

     about

     myself

     without

     compromising

     the

     integrity

     of

     our

     interaction

    .

     Please

     let

     me

     know

     if

     you

    'd

     like

     to

     proceed

     with

     the

     conversation

    .

     Thank

     you

    .

     [

    Name

    ]

     [

    Date

    ]

     [

    Your

     profession

    ,

     role

    ,

     or

     background

    ]

     [

    Your

     name

    ]

     [

    Your

     occupation

    ,

     role

    ,

     or

     experience

    ]

     [

    Your

     experience

     or

     skills

    ]

     [

    Your

    
    
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

     It

     is

     the

     most

     populous

     city

     in

     France

     and

     the

     seat

     of

     the

     French

     government

    .

     The

     city

     is

     located

     on

     the

     Se

    ine

     River

     in

     the

     center

     of

     the

     country

     and

     is

     home

     to

     many

     of

     the

     country

    ’s

     cultural

     and

     historical

     landmarks

    .

     Paris

     is

     known

     for

     its

     beautiful

     architecture

    ,

     rich

     history

    ,

     and

     annual

     fashion

     and

     art

     f

    airs

    ,

     as

     well

     as

     for

     its

     annual

     romance

     festival

    ,

     Le

     Festival

     de

     la

     Ch

    anson

    .

     Paris

     is

     also

     home

     to

     many

     notable

     museums

     and

     theaters

    ,

     including

     the

     Lou

    vre

     Museum

    ,

     the

     National

     Theatre

    ,

     and

     the

     Th

    é

    â

    tre

     du

     Sole

    il

    .

     The

     city

     is

     also

     home

     to

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     some

     possible

     trends

     we

     can

     speculate

     on

    :
    


    1

    .

     Increased

     Use

     of

     AI

     in

     Healthcare

    :

     With

     the

     prevalence

     of

     AI

     in

     healthcare

    ,

     we

     are

     likely

     to

     see

     more

     robotic

     and

     AI

    -ass

    isted

     treatments

     for

     diseases

     like

     cancer

    ,

     Alzheimer

    's

    ,

     and

     Parkinson

    's

    .

     This

     will

     lead

     to

     more

     personalized

     and

     efficient

     care

     for

     patients

    ,

     reducing

     the

     costs

     associated

     with

     traditional

     medical

     treatments

    .
    


    2

    .

     Development

     of

     AI

     Ethics

    :

     There

     is

     a

     growing

     concern

     about

     the

     ethical

     implications

     of

     AI

    .

     As

     AI

     becomes

     more

     advanced

    ,

     there

     will

     be

     a

     need

     for

     more

     transparent

     and

     ethical

     design

     and

     development

     processes

    .

     We

     may

     see

     more

     AI

     that

     is

     designed

     with

     ethical

     considerations

     in

     mind

    ,

    



```python
llm.shutdown()
```
