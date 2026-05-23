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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.70it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.70it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.70it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.77it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.77it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.77it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.77it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.78it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.78it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.78it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.78it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:10,  4.78it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:05,  8.10it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:05,  8.10it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:05,  8.10it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:05,  8.10it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:05,  8.10it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:03, 11.91it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:03, 11.91it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:03, 11.91it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 11.91it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:03, 11.91it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 15.72it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 15.72it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 15.72it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 15.72it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 15.72it/s]

    Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 15.72it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:01, 20.76it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 25.51it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 25.51it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 25.51it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 25.51it/s]

    Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 25.51it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 25.51it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 33.50it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 33.50it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 33.50it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 33.50it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 33.50it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 33.50it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 36.73it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 36.73it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 36.73it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 36.73it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 36.73it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 36.73it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 38.85it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 38.85it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 38.85it/s]

    Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 38.85it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 38.85it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 38.85it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 38.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 42.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.32 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.32 GB):   2%|▏         | 1/58 [00:00<00:08,  6.87it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.28 GB):   2%|▏         | 1/58 [00:00<00:08,  6.87it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=59.28 GB):   3%|▎         | 2/58 [00:00<00:07,  7.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.28 GB):   3%|▎         | 2/58 [00:00<00:07,  7.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.28 GB):   5%|▌         | 3/58 [00:00<00:07,  7.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.28 GB):   5%|▌         | 3/58 [00:00<00:07,  7.53it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.28 GB):   7%|▋         | 4/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.28 GB):   7%|▋         | 4/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.27 GB):   7%|▋         | 4/58 [00:00<00:06,  8.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.27 GB):  10%|█         | 6/58 [00:00<00:05,  9.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.26 GB):  10%|█         | 6/58 [00:00<00:05,  9.45it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=59.26 GB):  10%|█         | 6/58 [00:00<00:05,  9.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.26 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.26 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.25 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.25 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.25 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.47it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=59.25 GB):  17%|█▋        | 10/58 [00:01<00:03, 12.47it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.24 GB):  17%|█▋        | 10/58 [00:01<00:03, 12.47it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.24 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.24 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.24 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.23 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.62it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=59.23 GB):  28%|██▊       | 16/58 [00:01<00:02, 18.50it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.23 GB):  28%|██▊       | 16/58 [00:01<00:02, 18.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.22 GB):  28%|██▊       | 16/58 [00:01<00:02, 18.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.22 GB):  28%|██▊       | 16/58 [00:01<00:02, 18.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.22 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.22 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.38it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.20 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.38it/s]Capturing num tokens (num_tokens=960 avail_mem=59.21 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.38it/s] Capturing num tokens (num_tokens=896 avail_mem=59.21 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.38it/s]

    Capturing num tokens (num_tokens=896 avail_mem=59.21 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=832 avail_mem=59.21 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=768 avail_mem=59.20 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=704 avail_mem=59.20 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=640 avail_mem=59.20 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=576 avail_mem=59.20 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=576 avail_mem=59.20 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.50it/s]Capturing num tokens (num_tokens=512 avail_mem=59.18 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.50it/s]Capturing num tokens (num_tokens=480 avail_mem=59.20 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.50it/s]Capturing num tokens (num_tokens=448 avail_mem=59.20 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.50it/s]Capturing num tokens (num_tokens=416 avail_mem=59.19 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.50it/s]Capturing num tokens (num_tokens=384 avail_mem=59.19 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.50it/s]

    Capturing num tokens (num_tokens=384 avail_mem=59.19 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=352 avail_mem=59.19 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=320 avail_mem=59.18 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=288 avail_mem=59.18 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=256 avail_mem=59.18 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=240 avail_mem=59.17 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=240 avail_mem=59.17 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.70it/s]Capturing num tokens (num_tokens=224 avail_mem=59.17 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.70it/s]Capturing num tokens (num_tokens=208 avail_mem=58.03 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.70it/s]

    Capturing num tokens (num_tokens=192 avail_mem=58.03 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.70it/s]Capturing num tokens (num_tokens=176 avail_mem=58.03 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.70it/s]Capturing num tokens (num_tokens=160 avail_mem=59.13 GB):  66%|██████▌   | 38/58 [00:02<00:00, 39.70it/s]Capturing num tokens (num_tokens=160 avail_mem=59.13 GB):  74%|███████▍  | 43/58 [00:02<00:00, 25.57it/s]Capturing num tokens (num_tokens=144 avail_mem=59.12 GB):  74%|███████▍  | 43/58 [00:02<00:00, 25.57it/s]

    Capturing num tokens (num_tokens=128 avail_mem=58.13 GB):  74%|███████▍  | 43/58 [00:02<00:00, 25.57it/s]Capturing num tokens (num_tokens=112 avail_mem=58.13 GB):  74%|███████▍  | 43/58 [00:02<00:00, 25.57it/s]Capturing num tokens (num_tokens=96 avail_mem=58.13 GB):  74%|███████▍  | 43/58 [00:02<00:00, 25.57it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=58.13 GB):  81%|████████  | 47/58 [00:02<00:00, 20.41it/s]Capturing num tokens (num_tokens=80 avail_mem=59.11 GB):  81%|████████  | 47/58 [00:02<00:00, 20.41it/s]Capturing num tokens (num_tokens=64 avail_mem=58.18 GB):  81%|████████  | 47/58 [00:02<00:00, 20.41it/s]Capturing num tokens (num_tokens=48 avail_mem=58.18 GB):  81%|████████  | 47/58 [00:02<00:00, 20.41it/s]

    Capturing num tokens (num_tokens=48 avail_mem=58.18 GB):  86%|████████▌ | 50/58 [00:02<00:00, 17.86it/s]Capturing num tokens (num_tokens=32 avail_mem=58.18 GB):  86%|████████▌ | 50/58 [00:02<00:00, 17.86it/s]Capturing num tokens (num_tokens=28 avail_mem=59.09 GB):  86%|████████▌ | 50/58 [00:02<00:00, 17.86it/s]Capturing num tokens (num_tokens=24 avail_mem=58.24 GB):  86%|████████▌ | 50/58 [00:02<00:00, 17.86it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.24 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.29it/s]Capturing num tokens (num_tokens=20 avail_mem=58.23 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.29it/s]Capturing num tokens (num_tokens=16 avail_mem=59.09 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.29it/s]Capturing num tokens (num_tokens=16 avail_mem=59.09 GB):  95%|█████████▍| 55/58 [00:02<00:00, 16.00it/s]Capturing num tokens (num_tokens=12 avail_mem=58.30 GB):  95%|█████████▍| 55/58 [00:02<00:00, 16.00it/s]

    Capturing num tokens (num_tokens=8 avail_mem=58.29 GB):  95%|█████████▍| 55/58 [00:03<00:00, 16.00it/s] Capturing num tokens (num_tokens=8 avail_mem=58.29 GB):  98%|█████████▊| 57/58 [00:03<00:00, 15.21it/s]Capturing num tokens (num_tokens=4 avail_mem=59.08 GB):  98%|█████████▊| 57/58 [00:03<00:00, 15.21it/s]Capturing num tokens (num_tokens=4 avail_mem=59.08 GB): 100%|██████████| 58/58 [00:03<00:00, 18.20it/s]


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
    Generated text:  Steve. I am a junior at the University of California, Berkeley. I live in Oakland with my parents, my brother, and my best friend. I have a very healthy relationship with my family and my friends. I am a member of the Student Council of the University of California, Berkeley. When I was a little kid, I was very lucky to have my parents. I grew up in a very privileged family. The only difference was that my parents were both police officers. My parents were very strict about what I ate and how I was dressed. They made sure I was healthy and raised me to be a good person. But my
    ===============================
    Prompt: The president of the United States is
    Generated text:  a title held by a person with the title of President of the United States. The current president of the United States is Donald Trump. The current president was a candidate of the Democratic Party. His predecessor was former President Bill Clinton.
    Does this next sentence follow, given the preceding text?
    The predecessor of the president is a politician.
    
    Choose from: [i] yes [ii] no
    [i] yes
    
    The sentence "The predecessor of the president is a politician" logically follows from the preceding text because the text states that the current president of the United States is Donald Trump and the current president was a candidate of the Democratic Party. Since
    ===============================
    Prompt: The capital of France is
    Generated text:  ( ).
    A. Paris
    B. Lyon
    C. Bordeaux
    D. Toulouse
    Answer:
    
    A
    
    The most likely diagnosis for this condition is ____. 
    A. Acute gastroenteritis
    B. Viral hepatitis
    C. Acute pyelonephritis
    D. Toxic bacillary dysentery
    E. Bacterial dysentery
    Answer:
    
    C
    
    The opening and closing of an Excel worksheet are controlled by which of the following options?
    A. Keyboard
    B. Mouse
    C. Worksheet tab
    D. Window title bar
    Answer:
    
    B
    
    Which of the following can be directly indicated by
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it's going to change the way we live. What will it do to our lives? Will it change the way we interact with each other? Will it change the way we work? Will it change the way we learn? Will it change the way we communicate?
    The future of AI is uncertain, but there is no doubt that it will change the way we live. It is possible that it will lead to the creation of a world where artificial intelligence becomes more intelligent than humans. However, it is also possible that it will lead to the creation of a world where humans become more intelligent than artificial intelligence.
    The impact of AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third largest city in the world by population. The city is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for business, finance, and entertainment. Paris is a popular tourist destination and a major cultural hub in Europe. The city is known for its fashion, art, and cuisine, and is a major economic and political center in France. The city is also home to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to improve, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will likely lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need for privacy and security. This will likely lead to new regulations and standards for AI development and use
    


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
    Generated text:  [Name] and I'm a [age] year old girl who has been [occupation] for [number of years]. I have a wide range of interests and hobbies, including [list your hobbies here]. I am always ready to learn and I enjoy [list your hobbies here]. I am a [type of person] and I am always looking for new ways to [describe your goals or interests]. I am [type of person] and I am always looking for new ways to [describe your goals or interests]. I am [type of person] and I am always looking for new ways to [describe your goals or interests]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris is the capital city of France, located on the Seine River in the center of the country. It's known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also home to the Louvre Museum, where many famous art pieces are housed. It's a major city with a rich history, fashion industry, and numerous museums and attractions. Paris is the second-largest city in France after Paris, with an estimated population of over 1.2 million people. It's a beautiful city with a vibrant culture, art, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and variable. However, some possible trends that could come to fruition over the next decade include:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to work alongside humans in areas such as language, decision-making, and creativity. This integration could lead to more efficient and effective solutions to complex problems.
    
    2. Greater use of AI in healthcare: AI could be used to improve patient care and outcomes in a variety of fields, including diagnosing diseases, predicting disease progression, and developing personalized treatment plans. This could lead to more effective treatments and better patient outcomes.
    
    3. Increased use of


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

     Sarah

    ,

     and

     I

    'm

     an

     intro

    verted

     mathematic

    ian

     at

     the

     University

     of

     Phoenix

    .

     I

     enjoy

     solving

     puzzles

     and

     engaging

     in

     conversation

    .

     If

     you

     need

     help

     with

     a

     math

     problem

    ,

     I

    'm

     always

     ready

     to

     assist

    .

     What

     kind

     of

     project

     is

     this

     math

     class

     focused

     on

    ?

     I

     enjoy

     discussing

     with

     you

     the

     beauty

     of

     math

    .

     Let

    's

     learn

     together

    !

     That

     sounds

     like

     a

     great

     project

    .

     I

    'm

     here

     to

     help

     with

     any

     questions

     or

     problems

     you

     might

     have

     in

     the

     math

     class

    .

     Let

    's

     do

     this

    !

     How

     about

     the

     future

     of

     math

     in

     the

     world

    ?

     Math

     has

     been

     a

     vital

     part

     of

     many

     of

     our

     daily

     lives

    .

     From

     navigation

     to

     finance

    ,

     it

    's

     impossible

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Se

    ine

     River

    ,

     with

     a

     population

     of

     over

     

    2

    .

    5

     million

     people

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     of

     France

     and

     is

     the

     largest

     city

     in

     the

     country

     by

     population

    .

     It

    's

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

     The

     city

     is

     home

     to

     many

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

     also

     known

     for

     its

     fashion

     industry

     and

     its

     contribution

     to

     the

     world

     of

     art

    ,

     particularly

     in

     painting

     and

     sculpture

    .

     With

     its

     blend

     of

     old

    -world

     charm

     and

     modern

    ity

    ,

     Paris

     continues

     to

     be

     a

     city

     of

     incredible

     beauty

     and

     culture

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     changing

    ,

     and

     it

     is

     difficult

     to

     predict

     exactly

     what

     will

     happen

    .

     However

    ,

     here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     AI

     in

     the

     coming

     years

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

     more

     people

     become

     aware

     of

     the

     negative

     impact

     of

     AI

     on

     society

    ,

     there

     is

     a

     growing

     push

     towards

     developing

     AI

     that

     is

     more

     ethical

     and

     sustainable

    .

     This

     could

     mean

     implementing

     more

     ethical

     AI

     policies

    ,

     promoting

     transparency

     and

     accountability

     in

     AI

     development

    ,

     and

     designing

     AI

     systems

     that

     can

     be

     used

     for

     good

    .
    


    2

    .

     Advances

     in

     machine

     learning

     and

     deep

     learning

    :

     Machine

     learning

     and

     deep

     learning

     are

     two

     of

     the

     most

     powerful

     areas

     of

     AI

     research

    ,

     and

     they

     are

     likely

     to

    



```python
llm.shutdown()
```
