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


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-22 21:17:22] `torch_dtype` is deprecated! Use `dtype` instead!
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.36it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.36it/s]


    2026-04-22 21:17:26,957 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 21:17:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:04,  1.16s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:04,  1.16s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:04,  1.16s/it]

    Compiling num tokens (num_tokens=6144):   3%|▎         | 2/58 [00:02<01:04,  1.16s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:19,  2.76it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:19,  2.76it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:19,  2.76it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:19,  2.76it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:09,  5.10it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:09,  5.10it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:09,  5.10it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:09,  5.10it/s]

    Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:03<00:09,  5.10it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:03<00:09,  5.10it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:04,  9.87it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:04,  9.87it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:04,  9.87it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:04,  9.87it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:04,  9.87it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:03<00:04,  9.87it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:03<00:04,  9.87it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 16.36it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 16.36it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 16.36it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 16.36it/s] 

    Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 16.36it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 16.36it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 16.36it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 16.36it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 24.00it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 24.00it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 24.00it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 24.00it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 24.00it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 24.00it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 24.00it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:00, 30.24it/s]

    Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 38.07it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 38.07it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 38.07it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 38.07it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 38.07it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 38.07it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 38.07it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 38.07it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 43.72it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 43.72it/s] 

    Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 43.72it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 43.72it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 43.72it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 43.72it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 43.72it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 43.72it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 49.35it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 49.35it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 49.35it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 49.35it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 49.35it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 49.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.21 GB):   3%|▎         | 2/58 [00:00<00:04, 13.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.19 GB):   3%|▎         | 2/58 [00:00<00:04, 13.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.20 GB):   3%|▎         | 2/58 [00:00<00:04, 13.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.20 GB):   7%|▋         | 4/58 [00:00<00:03, 15.71it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.20 GB):   7%|▋         | 4/58 [00:00<00:03, 15.71it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.19 GB):   7%|▋         | 4/58 [00:00<00:03, 15.71it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.19 GB):  10%|█         | 6/58 [00:00<00:03, 17.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.19 GB):  10%|█         | 6/58 [00:00<00:03, 17.33it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=59.18 GB):  10%|█         | 6/58 [00:00<00:03, 17.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.17 GB):  10%|█         | 6/58 [00:00<00:03, 17.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.17 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.17 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.16 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.15 GB):  21%|██        | 12/58 [00:00<00:02, 22.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.15 GB):  21%|██        | 12/58 [00:00<00:02, 22.81it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=59.15 GB):  21%|██        | 12/58 [00:00<00:02, 22.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.14 GB):  21%|██        | 12/58 [00:00<00:02, 22.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.14 GB):  21%|██        | 12/58 [00:00<00:02, 22.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.14 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.14 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.12 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.10 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=960 avail_mem=59.12 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.81it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=59.12 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=832 avail_mem=59.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=768 avail_mem=59.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=704 avail_mem=59.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=704 avail_mem=59.11 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.38it/s]Capturing num tokens (num_tokens=640 avail_mem=59.10 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.38it/s]Capturing num tokens (num_tokens=576 avail_mem=59.10 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.38it/s]Capturing num tokens (num_tokens=512 avail_mem=59.09 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.38it/s]Capturing num tokens (num_tokens=480 avail_mem=59.11 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.38it/s]Capturing num tokens (num_tokens=448 avail_mem=59.10 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.38it/s]Capturing num tokens (num_tokens=448 avail_mem=59.10 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=416 avail_mem=59.10 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.67it/s]

    Capturing num tokens (num_tokens=384 avail_mem=59.10 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=352 avail_mem=59.09 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=320 avail_mem=59.09 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=288 avail_mem=59.09 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=288 avail_mem=59.09 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=256 avail_mem=59.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=240 avail_mem=59.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=224 avail_mem=59.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=208 avail_mem=59.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=192 avail_mem=59.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.82it/s]Capturing num tokens (num_tokens=192 avail_mem=59.08 GB):  71%|███████   | 41/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=176 avail_mem=59.07 GB):  71%|███████   | 41/58 [00:01<00:00, 45.59it/s]

    Capturing num tokens (num_tokens=160 avail_mem=59.07 GB):  71%|███████   | 41/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=144 avail_mem=59.06 GB):  71%|███████   | 41/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=128 avail_mem=59.06 GB):  71%|███████   | 41/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=112 avail_mem=59.06 GB):  71%|███████   | 41/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=112 avail_mem=59.06 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.40it/s]Capturing num tokens (num_tokens=96 avail_mem=59.06 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.40it/s] Capturing num tokens (num_tokens=80 avail_mem=59.05 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.40it/s]Capturing num tokens (num_tokens=64 avail_mem=59.05 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.40it/s]Capturing num tokens (num_tokens=48 avail_mem=59.05 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.40it/s]Capturing num tokens (num_tokens=32 avail_mem=59.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.40it/s]Capturing num tokens (num_tokens=32 avail_mem=59.04 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=28 avail_mem=59.04 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]

    Capturing num tokens (num_tokens=24 avail_mem=59.03 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=20 avail_mem=59.03 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=16 avail_mem=59.03 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=12 avail_mem=59.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=12 avail_mem=59.02 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=8 avail_mem=59.02 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.47it/s] Capturing num tokens (num_tokens=4 avail_mem=59.02 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=4 avail_mem=59.02 GB): 100%|██████████| 58/58 [00:01<00:00, 37.09it/s]


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
    Generated text:  Xander. I'm a high school student at the time. I went to college to get a degree in computer science and my dream is to become a professional programmer. I currently work as a software engineer and develop web applications for startups and large companies. I always love to think outside the box and have an adventurous spirit. I enjoy reading, traveling, and spending time with my family. What advice would you give to someone who is considering becoming a programmer?
    
    As an AI language model, I don't have personal experiences or feelings, but I can provide some general advice for someone considering a career in programming:
    
    1. Identify your passion:
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. This is a correct statement. However, if we consider the statement about the president of the United States to be true, it is possible that the president is also a person and a person is a type of thing (an object), and therefore, it is possible to be a person and a person is also a type of thing. Is the following statement true or false: If "A person is a person" is true, then "A person is a type of thing" is true? 
    
    1. False
    2. True
    3. Cannot be determined
    4. Not enough information to determine
    
    To determine the correct
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Marseille
    C. Lyon
    D. Toulouse
    Answer: A
    
    The Yijiang District is the _____ of Hangzhou.
    A. Financial District
    B. Commercial District
    C. Residential District
    D. Industrial District
    Answer: A
    
    The early period of China's socialist transformation was a transition from public ownership of means of production to the unity of public ownership and the distribution of ownership. The starting point of this transition is ____
    A. The proposal of the General Line for the Transition Period
    B. The adoption of the first Five-Year Plan
    C. The completion of the
    ===============================
    Prompt: The future of AI is
    Generated text:  a futuristic future, and it will redefine how we live, work, and play.
    The future of AI is one that is governed by the principles of transparency, accountability, and ethics. AI systems must be designed to be transparent to users and developers, ensuring that their decisions and actions are understandable and accountable. Additionally, AI systems should be held accountable for their actions, with the ability to be held responsible if they cause harm or unintended consequences. In addition, ethical guidelines should be established to guide the development and deployment of AI systems, ensuring that they are used in ways that are beneficial and equitable.
    AI has the potential to significantly impact many aspects


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title] at [company name]. I'm always looking for ways to [job title] and [job title] at [company name], and I'm always eager to learn and grow. I'm a [job title] at [company name], and I'm always looking for ways to [job title] and [job title] at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling city with a diverse population and is home to many famous artists, writers, and musicians. It is a popular tourist destination and a major economic center in Europe. The city is also known for its cuisine, including French cuisine, and is home to many museums, theaters, and other cultural institutions. Paris is a vibrant and dynamic city that continues to be a major cultural and economic center in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we are likely to see an increase in automation and robotics in various industries, from manufacturing to healthcare. This will lead to more efficient and cost-effective solutions, but it will also create new jobs and raise ethical concerns.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing pressure to address ethical concerns and privacy issues. This will require a more nuanced approach to AI development
    


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
    Generated text:  [Name] and I'm [Age]. I've been working in the field of [field of work] for [number of years] years. I have a passion for [reason why you like your work]. I also have a keen interest in [why you enjoy your hobbies] and I enjoy spending [time on hobbies] with family or friends. I'm a [job description] who is always looking for ways to improve my skills and expand my knowledge. I'm passionate about [main passion], and I'm always learning and evolving. I strive to be a positive role model for those around me. I'm always looking for new
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the central region of the country. It is the largest city in France, with a population of over 1. 3 million people. It is also one of the oldest cities in the world, having existed since the 6th century. Paris is known for its rich cultural heritage, including its renowned museums, landmarks, and opera houses. It is a cosmopolitan city with a diverse population of people from all over the world. The city's art and music scene is renowned throughout the world, and its fashion industry is highly regarded. Paris is known for its fashion, art, and historical significance, making it a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  not only exciting but also complex, with multiple trends and potential applications. Here are some potential future trends in AI:
    
    1. Advancements in neural networks: Neural networks, a type of machine learning algorithm, are likely to become even more powerful in the future, leading to even more accurate and sophisticated AI systems.
    
    2. Integration with other technologies: AI is increasingly being integrated with other technologies such as blockchain, Internet of Things (IoT), and machine learning, creating a more integrated and interconnected system.
    
    3. Increased focus on privacy and security: As AI systems become more complex and powerful, there is a growing emphasis on ensuring privacy and security


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

     first

     name

    ],

     and

     I

     am

     [

    insert

     profession

    ].

     I

     have

     always

     loved

     learning

     and

     have

     a

     passion

     for

     [

    insert

     a

     relevant

     field

    ],

     and

     I

     am

     always

     looking

     to

     expand

     my

     knowledge

    .

     I

     am

     passionate

     about

     [

    insert

     an

     experience

     or

     hobby

    ],

     and

     I

     am

     always

     eager

     to

     share

     my

     knowledge

     with

     others

    .

     I

     am

     a

     [

    insert

     a

     brief

     description

     of

     your

     character

    ].

     I

     hope

     to

     be

     able

     to

     help

     you

     in

     whatever

     way

     I

     can

    ,

     and

     I

     am

     always

     willing

     to

     learn

     and

     grow

    .

     So

    ,

     please

     feel

     free

     to

     ask

     me

     anything

    ,

     and

     I

     will

     do

     my

     best

     to

     answer

     you

    .

     [

    insert

     your

     character

    's

     name

    ]

     [

    insert

     any

     additional

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    The

     statement

     provides

     a

     concise

     factual

     statement

     about

     Paris

    ,

     the

     capital

     city

     of

     France

    .

     The

     capital

     city

     of

     France

     is

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     be

     a

     complex

     and

     rapidly

     evolving

     field

    .

     Some

     possible

     trends

     that

     may

     emerge

     in

     the

     coming

     years

     include

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

     we

     continue

     to

     develop

     AI

    ,

     it

     is

     likely

     to

     become

     more

     aligned

     with

     human

     values

     and

     ethical

     principles

    .

     As

     such

    ,

     there

     may

     be

     greater

     emphasis

     on

     developing

     AI

     that

     is

     designed

     to

     be

     transparent

    ,

     accountable

    ,

     and

     responsible

    .
    


    2

    .

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    :

     AI

     is

     already

     becoming

     more

     integrated

     into

     our

     daily

     lives

    ,

     from

     smart

     homes

     to

     voice

     assistants

    .

     It

     is

     likely

     that

     this

     trend

     will

     continue

    ,

     with

     AI

     becoming

     even

     more

     integrated

     into

     our

     everyday

     routines

    .
    


    3

    .

     AI

    



```python
llm.shutdown()
```
