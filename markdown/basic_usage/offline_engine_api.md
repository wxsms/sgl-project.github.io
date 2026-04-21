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
    [2026-04-21 05:30:35] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.52it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.51it/s]


    2026-04-21 05:30:39,991 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 05:30:39] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:38,  2.77s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:10,  4.87it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:10,  4.87it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:10,  4.87it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:10,  4.87it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:10,  4.87it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:03<00:10,  4.87it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:03<00:10,  4.87it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:03<00:10,  4.87it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:04, 10.31it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:04, 10.31it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:04, 10.31it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:04, 10.31it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:04, 10.31it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:03<00:04, 10.31it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:03<00:04, 10.31it/s] 

    Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:03<00:04, 10.31it/s]Compiling num tokens (num_tokens=832):  28%|██▊       | 16/58 [00:03<00:04, 10.31it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 17.62it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 17.62it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 17.62it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 17.62it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 17.62it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 17.62it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:01, 17.62it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:03<00:01, 17.62it/s]Compiling num tokens (num_tokens=416):  41%|████▏     | 24/58 [00:03<00:01, 17.62it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 25.70it/s]

    Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:03<00:01, 25.70it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 33.94it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 33.94it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 33.94it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 33.94it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 33.94it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 33.94it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 33.94it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 33.94it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:03<00:00, 33.94it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 41.65it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 41.65it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 41.65it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 41.65it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 41.65it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 41.65it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 41.65it/s]

    Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 41.65it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 41.65it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 41.65it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 50.66it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 50.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=117.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=117.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=117.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=117.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.31it/s]Capturing num tokens (num_tokens=5632 avail_mem=117.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.31it/s]Capturing num tokens (num_tokens=5120 avail_mem=117.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=117.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=117.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=117.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=117.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=117.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=117.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=117.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=117.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=117.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.36it/s]Capturing num tokens (num_tokens=2560 avail_mem=117.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.36it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.36it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=117.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=117.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=117.04 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.14it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=117.02 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=117.02 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=960 avail_mem=117.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.85it/s] Capturing num tokens (num_tokens=896 avail_mem=117.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=832 avail_mem=117.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=768 avail_mem=117.02 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=704 avail_mem=117.02 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.85it/s]Capturing num tokens (num_tokens=704 avail_mem=117.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.27it/s]Capturing num tokens (num_tokens=640 avail_mem=117.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.27it/s]Capturing num tokens (num_tokens=576 avail_mem=117.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.27it/s]Capturing num tokens (num_tokens=512 avail_mem=117.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.27it/s]

    Capturing num tokens (num_tokens=480 avail_mem=117.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.27it/s]Capturing num tokens (num_tokens=448 avail_mem=117.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.27it/s]Capturing num tokens (num_tokens=448 avail_mem=117.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=416 avail_mem=117.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=384 avail_mem=117.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=352 avail_mem=117.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=320 avail_mem=117.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=288 avail_mem=117.00 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=288 avail_mem=117.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.85it/s]Capturing num tokens (num_tokens=256 avail_mem=117.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.85it/s]Capturing num tokens (num_tokens=240 avail_mem=117.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.85it/s]

    Capturing num tokens (num_tokens=224 avail_mem=117.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.85it/s]Capturing num tokens (num_tokens=208 avail_mem=116.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.85it/s]Capturing num tokens (num_tokens=192 avail_mem=116.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.85it/s]Capturing num tokens (num_tokens=192 avail_mem=116.99 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=176 avail_mem=116.99 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=160 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=144 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=128 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=112 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=112 avail_mem=116.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=96 avail_mem=116.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.99it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=116.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=64 avail_mem=116.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=48 avail_mem=116.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=32 avail_mem=116.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=32 avail_mem=116.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.89it/s]Capturing num tokens (num_tokens=28 avail_mem=116.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.89it/s]Capturing num tokens (num_tokens=24 avail_mem=116.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.89it/s]Capturing num tokens (num_tokens=20 avail_mem=116.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.89it/s]Capturing num tokens (num_tokens=16 avail_mem=116.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.89it/s]Capturing num tokens (num_tokens=12 avail_mem=116.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.89it/s]

    Capturing num tokens (num_tokens=12 avail_mem=116.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=8 avail_mem=116.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.43it/s] Capturing num tokens (num_tokens=4 avail_mem=116.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=4 avail_mem=116.94 GB): 100%|██████████| 58/58 [00:01<00:00, 37.80it/s]


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
    Generated text:  Naima. I have been looking for a new job for quite a while. As I have been trying to find a position, I have been told a few things. First, the new position is a position in the Government of Saudi Arabia. Second, the position involves a lot of administrative work. Third, the pay rate is not very high. After I talk to people who have tried to find jobs in the Government of Saudi Arabia, they have told me that the work is very difficult. Now, my question is, how do I go about finding a job in the Government of Saudi Arabia? (with English)
    
    Certainly, I
    ===============================
    Prompt: The president of the United States is
    Generated text:  facing a situation where the debt ceiling needs to be raised to ensure the government remains solvent. The debt ceiling is a cap on the amount of money that can be borrowed by the government.
    
    The president has three options:
    
    1. Raise the debt ceiling by $100 million immediately.
    2. Raise the debt ceiling by $200 million immediately, and then $800 million per year for 10 years.
    3. Increase the debt ceiling by $300 million immediately, and then $600 million per year for 10 years.
    
    Assuming that the government's debt is currently $10 trillion
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. When you visit Paris, it’s a must-see destination. Here are some of the most famous landmarks in Paris.
    The Louvre Museum, also known as the National Museum of France, is the largest museum in the world. It contains the world’s largest collection of art and antiquities. The museum contains more than 600,000 objects in its collection.
    Piccadilly Circus is a huge city center square, full of world famous landmarks such as the Eiffel Tower, the Houses of Parliament, and the British Museum.
    The Louvre Museum, also known as the National Museum of France, is
    ===============================
    Prompt: The future of AI is
    Generated text:  predictably uncertain. As the number of AI models rapidly grows, developers will need to be constantly adapting to new techniques to keep up. One promising technique is the use of machine learning (ML) in the form of Deep Learning, a type of machine learning that uses complex neural networks to perform tasks. Deep learning algorithms can learn to recognize patterns and make predictions based on large amounts of data, making them an attractive option for many industries.
    One major challenge with Deep Learning is its ability to handle large amounts of data. Traditional machine learning techniques can be limited by the amount of data they need to process, which can lead to issues such as over


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? I'm a [insert a short description of your personality or skills]. And what's your favorite hobby or activity? I love [insert a hobby or activity you enjoy]. And what's your favorite book or movie? I love [insert a book or movie you've read or watched]. And what's your favorite place to relax? I love [insert a place you enjoy]. And what's your favorite color? I love [insert a favorite color
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French Quarter, where many famous French artists and writers have lived and worked. Paris is a bustling city with a rich history and culture, and it is a popular tourist destination. The city is also known for its cuisine, including its famous French fries and its famous cheese, brie. Paris is a city that is both a cultural and historical center of France, and it is a must-visit destination for anyone interested in French culture and history. 
    
    Therefore, the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation: One of the most significant trends in AI is the increasing automation of tasks that are currently performed by humans. This could include tasks such as data analysis, decision-making, and problem-solving, as well as tasks that are currently performed by machines.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there is a risk that they may be used to collect and analyze personal data without the consent of the
    


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
    Generated text:  [insert character's name] and I'm a [insert fictional occupation or profession] with [insert a bit about your background]. I enjoy [insert a few things about your background], and I'm always looking to learn new things. Whether it's coding, cooking, or anything else, I'm always eager to expand my knowledge. What's your background? What's your profession? What's your hobbies? What's your favorite thing to do? How do you handle stress? What's your favorite movie or book? And what's your dream job? I'm always open to learning more about you, so please feel free to ask me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the capital city of France and is known for its iconic architecture, rich history, and vibrant culture. It is the largest city in France and serves as the administrative and cultural center of the country. The city is home to many world-famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its cuisine and is the birthplace of many world-renowned artists such as Van Gogh and Picasso. As a result, Paris is one of the most popular and economically important cities in Europe. 
    Therefore, the answer is:
    
    Paris is the capital city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting, and there are many potential trends that could shape the technology's direction. Here are some potential future trends in AI:
    
    1. Improved Natural Language Processing: As AI continues to advance, we can expect to see improvements in natural language processing, which will allow machines to understand and interpret human language more accurately and effectively.
    
    2. Greater Integration of AI with other technologies: AI is already integrating with other technologies, such as sensors, cameras, and drones, but there are also potential opportunities for even more seamless integration with other systems.
    
    3. Increased Use of AI in Healthcare: AI is already being used in healthcare to help diagnose diseases, predict


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

     Emily

    .

     I

    'm

     a

     young

     woman

     with

     an

     artistic

     heart

     and

     a

     passion

     for

     writing

    .

     I

    'm

     a

     curious

     person

     who

     enjoys

     exploring

     new

     stories

     and

     worlds

    ,

     and

     I

     find

     myself

     drawn

     to

     the

     challenge

     of

     coming

     up

     with

     unique

     and

     imaginative

     worlds

    .

     I

     love

     to

     write

     in

     the

     form

     of fiction

     and

     am

     excited

     about

     sharing

     my

     ideas

     with

     others

    .

     I

    'm

     always

     up

     for

     a

     challenge

     and

     eager

     to

     learn

     new

     things

    .

     I

    'm

     looking

     forward

     to

     meeting

     you

    .

     How

     about

     you

    ?

     Is

     there

     anything

     specific

     you

    'd

     like

     to

     share

     about

     yourself

    ?

     How

     about

     your

     love

     of

     writing

    ?

     Is

     there

     anything

     specific

     you

    'd

     like

     to

     tell

     me

     about

     yourself

    ?


    Sure

    ,

     I

    'd

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Par

    ís

    ".

     Paris

     is

     the

     largest

     city

     in

     the

     European

     Union

     and

     is

     home

     to

     the

     French

     government

    ,

     the

     French

     Parliament

    ,

     and

     the

     European

     Parliament

    .

     The

     city

     is

     also

     home

     to

     many

     of

     France

    's

     most

     famous

     landmarks

    ,

     including

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

    .

     Paris

     is

     known

     for

     its

     classical

     architecture

    ,

     vibrant

     music

     scene

    ,

     and

     diverse

     cuisine

    .

     It

     is

     a

     global

     city

     with

     a

     rich

     cultural

     history

     and

     a

     thriving

     economy

    .

     The

     city

     has

     a

     population

     of

     over

     

    2

    .

     

    2

     million

     people

     and

     has

     been

     the

     capital

     of

     France

     since

     

    1

    8

    0

    4

    .

     It

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     complex

     and

     uncertain

    ,

     but

     there

     are

     several

     possible

     trends

     that

     may

     shape

     its

     development

    .

     Here

     are

     some

     potential

     areas

     of

     focus

    :
    


    1

    .

     Artificial

     general

     intelligence

    :

     This

     is

     the

     goal

     of

     creating

     machines

     that

     can

     perform

     any

     task

     that

     a

     human

     can

     do

    ,

     without

     being

     explicitly

     programmed

     to

     do

     so

    .

     This

     is

     a

     challenging

     goal

    ,

     and researchers

     are

     working

     on

     developing

     algorithms

     and

     systems

     that

     can

     approximate

     human

    -level

     performance

    .
    


    2

    .

     Narrow

     AI

    :

     Narrow

     AI

     is

     a

     subset

     of

     AI

     that

     focuses

     on

     specific

     tasks

    ,

     such

     as

     image

     recognition

     or

     language

     translation

    .

     Researchers

     are

     developing

     new

     algorithms

     and

     models

     to

     make

     Narrow

     AI

     more

     effective

     and

     efficient

    .
    


    3

    .

     Autonomous

     vehicles

    :

     AI

     is

     transforming

    



```python
llm.shutdown()
```
