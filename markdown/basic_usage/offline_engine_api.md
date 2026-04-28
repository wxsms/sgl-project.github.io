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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.37it/s]


    2026-04-28 23:01:51,286 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 23:01:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:10,  1.27s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:10,  1.27s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:10,  1.27s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:10,  1.27s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:27,  1.92it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:27,  1.92it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:05<00:27,  1.92it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:05<00:27,  1.92it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:14,  3.41it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:14,  3.41it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:05<00:14,  3.41it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:05<00:14,  3.41it/s]

    Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:05<00:14,  3.41it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:07,  5.98it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:07,  5.98it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:07,  5.98it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:07,  5.98it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:07,  5.98it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:05<00:07,  5.98it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:05<00:07,  5.98it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:03, 10.85it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:03, 10.85it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:03, 10.85it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:03, 10.85it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:03, 10.85it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:03, 10.85it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:03, 10.85it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:03, 10.85it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 17.47it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 17.47it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 17.47it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 17.47it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 17.47it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:01, 17.47it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:01, 17.47it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:01, 17.47it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:05<00:01, 17.47it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 26.14it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 26.14it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 26.14it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 26.14it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 26.14it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:00, 26.14it/s]

    Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:00, 26.14it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:00, 26.14it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 33.33it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 33.33it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 33.33it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 33.33it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 33.33it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 33.33it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 33.33it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 33.33it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 33.33it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 41.25it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 41.25it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 41.25it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 41.25it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 41.25it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 41.25it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 41.25it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 41.25it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 41.25it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 41.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=112.92 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=112.92 GB):   2%|▏         | 1/58 [00:00<00:06,  8.24it/s]Capturing num tokens (num_tokens=7680 avail_mem=113.34 GB):   2%|▏         | 1/58 [00:00<00:06,  8.24it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=113.34 GB):   3%|▎         | 2/58 [00:00<00:06,  8.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=113.10 GB):   3%|▎         | 2/58 [00:00<00:06,  8.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=112.94 GB):   3%|▎         | 2/58 [00:00<00:06,  8.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=112.94 GB):   7%|▋         | 4/58 [00:00<00:05, 10.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=113.34 GB):   7%|▋         | 4/58 [00:00<00:05, 10.51it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=113.33 GB):   7%|▋         | 4/58 [00:00<00:05, 10.51it/s]Capturing num tokens (num_tokens=5632 avail_mem=113.33 GB):  10%|█         | 6/58 [00:00<00:04, 11.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=112.98 GB):  10%|█         | 6/58 [00:00<00:04, 11.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=113.31 GB):  10%|█         | 6/58 [00:00<00:04, 11.48it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=113.31 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=113.30 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.51it/s]Capturing num tokens (num_tokens=3840 avail_mem=112.98 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.51it/s]Capturing num tokens (num_tokens=3840 avail_mem=112.98 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=113.20 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.73it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=113.23 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=113.23 GB):  21%|██        | 12/58 [00:00<00:03, 14.06it/s]Capturing num tokens (num_tokens=3072 avail_mem=113.22 GB):  21%|██        | 12/58 [00:00<00:03, 14.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=113.21 GB):  21%|██        | 12/58 [00:00<00:03, 14.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=113.21 GB):  21%|██        | 12/58 [00:01<00:03, 14.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=113.21 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=113.20 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.99it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=113.00 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=113.19 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=113.19 GB):  31%|███       | 18/58 [00:01<00:02, 19.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=113.18 GB):  31%|███       | 18/58 [00:01<00:02, 19.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=113.18 GB):  31%|███       | 18/58 [00:01<00:02, 19.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=113.15 GB):  31%|███       | 18/58 [00:01<00:02, 19.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=113.15 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.26it/s]Capturing num tokens (num_tokens=960 avail_mem=113.16 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.26it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=113.15 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.26it/s]Capturing num tokens (num_tokens=832 avail_mem=113.15 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.26it/s]Capturing num tokens (num_tokens=832 avail_mem=113.15 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.35it/s]Capturing num tokens (num_tokens=768 avail_mem=113.14 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.35it/s]Capturing num tokens (num_tokens=704 avail_mem=113.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.35it/s]Capturing num tokens (num_tokens=640 avail_mem=113.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.35it/s]Capturing num tokens (num_tokens=576 avail_mem=113.12 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.35it/s]Capturing num tokens (num_tokens=576 avail_mem=113.12 GB):  48%|████▊     | 28/58 [00:01<00:01, 25.97it/s]Capturing num tokens (num_tokens=512 avail_mem=113.07 GB):  48%|████▊     | 28/58 [00:01<00:01, 25.97it/s]

    Capturing num tokens (num_tokens=480 avail_mem=113.08 GB):  48%|████▊     | 28/58 [00:01<00:01, 25.97it/s]Capturing num tokens (num_tokens=448 avail_mem=113.07 GB):  48%|████▊     | 28/58 [00:01<00:01, 25.97it/s]Capturing num tokens (num_tokens=416 avail_mem=113.03 GB):  48%|████▊     | 28/58 [00:01<00:01, 25.97it/s]Capturing num tokens (num_tokens=416 avail_mem=113.03 GB):  55%|█████▌    | 32/58 [00:01<00:00, 27.93it/s]Capturing num tokens (num_tokens=384 avail_mem=113.06 GB):  55%|█████▌    | 32/58 [00:01<00:00, 27.93it/s]Capturing num tokens (num_tokens=352 avail_mem=113.05 GB):  55%|█████▌    | 32/58 [00:01<00:00, 27.93it/s]Capturing num tokens (num_tokens=320 avail_mem=113.04 GB):  55%|█████▌    | 32/58 [00:01<00:00, 27.93it/s]

    Capturing num tokens (num_tokens=320 avail_mem=113.04 GB):  60%|██████    | 35/58 [00:01<00:00, 27.10it/s]Capturing num tokens (num_tokens=288 avail_mem=113.04 GB):  60%|██████    | 35/58 [00:01<00:00, 27.10it/s]Capturing num tokens (num_tokens=256 avail_mem=113.03 GB):  60%|██████    | 35/58 [00:01<00:00, 27.10it/s]Capturing num tokens (num_tokens=240 avail_mem=113.02 GB):  60%|██████    | 35/58 [00:01<00:00, 27.10it/s]Capturing num tokens (num_tokens=240 avail_mem=113.02 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.15it/s]Capturing num tokens (num_tokens=224 avail_mem=113.01 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.15it/s]Capturing num tokens (num_tokens=208 avail_mem=113.01 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.15it/s]

    Capturing num tokens (num_tokens=192 avail_mem=113.01 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.15it/s]Capturing num tokens (num_tokens=192 avail_mem=113.01 GB):  71%|███████   | 41/58 [00:02<00:00, 25.50it/s]Capturing num tokens (num_tokens=176 avail_mem=113.01 GB):  71%|███████   | 41/58 [00:02<00:00, 25.50it/s]Capturing num tokens (num_tokens=160 avail_mem=112.98 GB):  71%|███████   | 41/58 [00:02<00:00, 25.50it/s]Capturing num tokens (num_tokens=144 avail_mem=112.97 GB):  71%|███████   | 41/58 [00:02<00:00, 25.50it/s]Capturing num tokens (num_tokens=144 avail_mem=112.97 GB):  76%|███████▌  | 44/58 [00:02<00:00, 25.20it/s]Capturing num tokens (num_tokens=128 avail_mem=112.97 GB):  76%|███████▌  | 44/58 [00:02<00:00, 25.20it/s]

    Capturing num tokens (num_tokens=112 avail_mem=112.98 GB):  76%|███████▌  | 44/58 [00:02<00:00, 25.20it/s]Capturing num tokens (num_tokens=96 avail_mem=112.97 GB):  76%|███████▌  | 44/58 [00:02<00:00, 25.20it/s] Capturing num tokens (num_tokens=96 avail_mem=112.97 GB):  81%|████████  | 47/58 [00:02<00:00, 24.68it/s]Capturing num tokens (num_tokens=80 avail_mem=112.95 GB):  81%|████████  | 47/58 [00:02<00:00, 24.68it/s]Capturing num tokens (num_tokens=64 avail_mem=112.94 GB):  81%|████████  | 47/58 [00:02<00:00, 24.68it/s]Capturing num tokens (num_tokens=48 avail_mem=112.95 GB):  81%|████████  | 47/58 [00:02<00:00, 24.68it/s]Capturing num tokens (num_tokens=48 avail_mem=112.95 GB):  86%|████████▌ | 50/58 [00:02<00:00, 25.06it/s]Capturing num tokens (num_tokens=32 avail_mem=112.94 GB):  86%|████████▌ | 50/58 [00:02<00:00, 25.06it/s]

    Capturing num tokens (num_tokens=28 avail_mem=112.94 GB):  86%|████████▌ | 50/58 [00:02<00:00, 25.06it/s]Capturing num tokens (num_tokens=24 avail_mem=112.93 GB):  86%|████████▌ | 50/58 [00:02<00:00, 25.06it/s]Capturing num tokens (num_tokens=24 avail_mem=112.93 GB):  91%|█████████▏| 53/58 [00:02<00:00, 26.05it/s]Capturing num tokens (num_tokens=20 avail_mem=112.93 GB):  91%|█████████▏| 53/58 [00:02<00:00, 26.05it/s]Capturing num tokens (num_tokens=16 avail_mem=112.92 GB):  91%|█████████▏| 53/58 [00:02<00:00, 26.05it/s]Capturing num tokens (num_tokens=12 avail_mem=112.92 GB):  91%|█████████▏| 53/58 [00:02<00:00, 26.05it/s]Capturing num tokens (num_tokens=8 avail_mem=112.91 GB):  91%|█████████▏| 53/58 [00:02<00:00, 26.05it/s] Capturing num tokens (num_tokens=8 avail_mem=112.91 GB):  98%|█████████▊| 57/58 [00:02<00:00, 29.27it/s]Capturing num tokens (num_tokens=4 avail_mem=112.88 GB):  98%|█████████▊| 57/58 [00:02<00:00, 29.27it/s]

    Capturing num tokens (num_tokens=4 avail_mem=112.88 GB): 100%|██████████| 58/58 [00:02<00:00, 21.89it/s]


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
    Generated text:  Timmy. I'm a bit of a fan of math, and I'd like to write a paper on my experiences. I need help with a few questions to get started. Can you help me out? 
    
    1. What is your favorite math subject? 
    2. What is your favorite math topic? 
    3. What is your favorite math resource? 
    4. What is your favorite math book? 
    5. What is your favorite math way of thinking? 
    6. What is your favorite math topic outside of math? 
    7. What is your favorite math idea? 
    8. What is your favorite math formula? 
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy. He has many important meetings. When he is not meeting with other important people, he likes to read. He is reading a book called "The Greatest American Presidents." He likes to read this book a lot because he thinks it can help him learn more about important people. The book is a good choice for him because he is very busy with so many important things. He has to take care of his family and his children, and he has to work and go to important meetings. He usually does not have time to read. When he is reading a book, he feels very relaxed and he enjoys reading. He is always interested
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Amsterdam
    D. Berlin
    Answer: A
    
    In a given population, if the null hypothesis is rejected, it suggests that the observed result is likely a result of chance. True or False?
    A. True
    B. False
    Answer: B
    
    Which of the following statements about the principles of the International Code for the Design of Chemical Safety Data Sheets (CSDS) is true?
    A. All information should be fully included in CSDS
    B. Information on dangerousness of substances can be omitted
    C. There is no limit to the extent of information in
    ===============================
    Prompt: The future of AI is
    Generated text:  highly dependent on the availability of reliable and accurate data. This is a critical issue in the field of artificial intelligence, and it has been a significant challenge for researchers and practitioners. One of the main challenges in AI data generation is the creation of high-quality and diverse datasets that can be used to train and test machine learning algorithms. These datasets must be large, comprehensive, and representative of the real-world data. They must also be easily accessible and accessible to the researchers and practitioners working on the project.
    The availability of high-quality and diverse datasets is a critical issue in the field of artificial intelligence. One of the main challenges in AI data generation is


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast who enjoys [mention a hobby or interest]. I'm always looking for new experiences and learning new things, and I'm always eager to share my knowledge with others. I'm a [Favorite Book, Movie, or Sport] lover who loves to read, watch movies, and play sports. I'm also a [Favorite Music Genre] fan who loves to listen to music. I'm always looking for new ways to express myself and I'm always eager to try new things. I'm a [Favorite Animal] lover
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is located on the Seine River and is the seat of government for the country. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a popular tourist destination and a major economic center in France. It is also a major center for the arts and culture industry. The city is home to many museums, theaters, and other cultural institutions. Paris is a city of contrasts, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and responsible AI. This could include developing AI
    


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
    Generated text: ... [insert character's name] and I am a/an [insert character's age or occupation] [insert character's occupation] and I am [insert character's profession]. My [insert character's role] is [insert character's role] and I am [insert character's role] [insert character's role]. I am [insert character's name] and I am [insert character's profession]. I have always been [insert character's personality trait] [insert character's personality trait] and I am [insert character's age or occupation] [insert character's age or occupation]. I am passionate about [insert character's passion or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France and is the largest and most populous city in the country. It is located in the south of the country and is the fifth largest city in the European Union. It is a major cultural and economic center and is known for its beautiful architecture, rich history, and annual festivals such as the Eiffel Tower celebrations. Paris is also a hub of business, with many multinational corporations headquartered there. The city is home to many world-renowned museums, art galleries, and other cultural institutions. It is a popular tourist destination and is known for its food, fashion, and entertainment industry. Paris is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and diverse, with a number of trends shaping its future trajectory. Some possible future trends in AI include:
    
    1. Increased customization and personalization: AI will continue to become more personal, allowing users to customize their experiences with the AI system based on their preferences and behaviors. This trend will likely lead to more personalized services, such as chatbots that provide tailored recommendations and personalized product suggestions.
    
    2. Advancements in hardware and software: AI will continue to gain faster and more powerful hardware, such as faster CPUs and GPUs, which will allow for more powerful and complex AI algorithms. As software continues to evolve, we may see more sophisticated


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

     am

     an

     [

    Age

    ]

     year

     old

     [

    Occup

    ation

    ].

     I

     work

     as

     a

     [

    Occup

    ation

    ]

     and

     I

     have

     been

     living

     in

     [

    City

    ]

     for

     [

    Number

     of

     years

    ].

     I

     have

     been

     studying

     at

     [

    School

    ]

     and

     I

     love

     [

    Favorite

     thing

     to

     eat

    ]

     and

     [

    Favorite

     animal

    ].

     I

     am

     [

    Age

    ]

     years

     old

     and

     I

     am

     always

     looking

     for

     a

     challenge

     or

     adventure

    .

     I

     am

     happy

     to

     learn

     something

     new

     or

     try

     out

     a

     new

     hobby

    .

     I

     am

     [

    Name

    ]

     and

     I

     am

     always

     ready

     to

     help

     and

     to

     learn

     from

     someone

    .

     How

     would

     you

     describe

     the

     person

     you

     are

     as

    ?

     I

     am

     [

    Name

    ]

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     most

     populous

     city

     in

     the

     country

     and

     is

     known

     for

     its

     historical

     significance

     and

     thriving

     culture

    .

     Paris

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     Notre

     Dame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    ,

     among

     others

    .

     It

     is

     also

     a

     center

     for

     arts

    ,

     literature

    ,

     and

     music

    ,

     and

     is

     a

     popular

     tourist

     destination

    .

     Despite

     the

     challenges

     of

     climate

     change

     and

     economic

     issues

    ,

     Paris

     remains

     a

     vital

     part

     of

     French

     culture

     and

     a

     major

     hub

     for

     international

     affairs

    .

     The

     city

     is

     also

     home

     to

     the

     Lou

    vre

     Museum

    ,

     the

     biggest

     art

     collection

     in

     the

     world

    ,

     and

     the

     E

    iff

    el

     Tower

    ,

     which

     stands

     as

     a

     symbol

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     evolve

    ,

     with

     new

     technologies

     and

     applications

     being

     developed

     at

     a

     rapid

     pace

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Improved

     natural

     language

     processing

    :

     As

     AI

     continues

     to

     become

     more

     powerful

    ,

     it

    's

     likely

     that

     we

     will

     see

     more

     advanced

     natural

     language

     processing

     capabilities

    .

     This

     could

     include

     the

     ability

     to

     understand

     and

     interpret

     human

     speech

    ,

     as

     well

     as

     the

     ability

     to

     generate

     natural

     language

     responses

    .
    


    2

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

     in

     healthcare

     to

     improve

     patient

     outcomes

     and

     reduce

     costs

    .

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     may

     see

     even

     greater

     adoption

     of

     this

     technology

     in

     healthcare

    ,

     with

     more

     sophisticated

     algorithms

     and

     machine

     learning

    



```python
llm.shutdown()
```
