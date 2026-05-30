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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.95it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.68it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.68it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.68it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.68it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:05,  8.19it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:03, 12.24it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:03, 12.24it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:03, 12.24it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:03, 12.24it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:03, 12.24it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 16.37it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 16.37it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 16.37it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 16.37it/s]

    Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:02, 16.37it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:02, 16.37it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:01, 21.74it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:01, 21.74it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:01, 21.74it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:01, 21.74it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:01, 21.74it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:01, 21.74it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:00, 27.13it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:00, 27.13it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:00, 27.13it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:00, 27.13it/s]

    Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:00, 27.13it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:00, 27.13it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 36.93it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 36.93it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 36.93it/s]

    Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 36.93it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 36.93it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 36.93it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 36.93it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 40.92it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 40.92it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 40.92it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 40.92it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 40.92it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 40.92it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 40.92it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 45.04it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 45.04it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 45.04it/s] 

    Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 45.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.79 GB):   2%|▏         | 1/58 [00:00<00:07,  7.92it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.75 GB):   2%|▏         | 1/58 [00:00<00:07,  7.92it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=57.75 GB):   3%|▎         | 2/58 [00:00<00:07,  7.49it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.75 GB):   3%|▎         | 2/58 [00:00<00:07,  7.49it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.75 GB):   5%|▌         | 3/58 [00:00<00:07,  7.69it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.75 GB):   5%|▌         | 3/58 [00:00<00:07,  7.69it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.75 GB):   7%|▋         | 4/58 [00:00<00:06,  8.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.75 GB):   7%|▋         | 4/58 [00:00<00:06,  8.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.74 GB):   7%|▋         | 4/58 [00:00<00:06,  8.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.74 GB):  10%|█         | 6/58 [00:00<00:04, 10.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.73 GB):  10%|█         | 6/58 [00:00<00:04, 10.61it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=57.73 GB):  10%|█         | 6/58 [00:00<00:04, 10.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.73 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.73 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.05it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.73 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.05it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=57.73 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.72 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.72 GB):  17%|█▋        | 10/58 [00:01<00:03, 12.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.72 GB):  21%|██        | 12/58 [00:01<00:03, 12.47it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.71 GB):  21%|██        | 12/58 [00:01<00:03, 12.47it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=57.71 GB):  21%|██        | 12/58 [00:01<00:03, 12.47it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.71 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.71 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.71 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.71 GB):  28%|██▊       | 16/58 [00:01<00:03, 13.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.70 GB):  28%|██▊       | 16/58 [00:01<00:03, 13.72it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=57.70 GB):  28%|██▊       | 16/58 [00:01<00:03, 13.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.70 GB):  31%|███       | 18/58 [00:01<00:02, 15.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.70 GB):  31%|███       | 18/58 [00:01<00:02, 15.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.69 GB):  31%|███       | 18/58 [00:01<00:02, 15.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.69 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.96it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.67 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.96it/s]

    Capturing num tokens (num_tokens=960 avail_mem=57.69 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.96it/s] Capturing num tokens (num_tokens=960 avail_mem=57.69 GB):  38%|███▊      | 22/58 [00:01<00:02, 16.57it/s]Capturing num tokens (num_tokens=896 avail_mem=57.69 GB):  38%|███▊      | 22/58 [00:01<00:02, 16.57it/s]Capturing num tokens (num_tokens=832 avail_mem=57.68 GB):  38%|███▊      | 22/58 [00:01<00:02, 16.57it/s]Capturing num tokens (num_tokens=832 avail_mem=57.68 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.43it/s]Capturing num tokens (num_tokens=768 avail_mem=57.68 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.43it/s]Capturing num tokens (num_tokens=704 avail_mem=57.68 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.43it/s]

    Capturing num tokens (num_tokens=640 avail_mem=57.67 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.43it/s]Capturing num tokens (num_tokens=640 avail_mem=57.67 GB):  47%|████▋     | 27/58 [00:01<00:01, 18.88it/s]Capturing num tokens (num_tokens=576 avail_mem=57.67 GB):  47%|████▋     | 27/58 [00:01<00:01, 18.88it/s]Capturing num tokens (num_tokens=512 avail_mem=57.66 GB):  47%|████▋     | 27/58 [00:01<00:01, 18.88it/s]Capturing num tokens (num_tokens=512 avail_mem=57.66 GB):  50%|█████     | 29/58 [00:02<00:01, 19.14it/s]Capturing num tokens (num_tokens=480 avail_mem=56.57 GB):  50%|█████     | 29/58 [00:02<00:01, 19.14it/s]Capturing num tokens (num_tokens=448 avail_mem=56.57 GB):  50%|█████     | 29/58 [00:02<00:01, 19.14it/s]

    Capturing num tokens (num_tokens=416 avail_mem=56.57 GB):  50%|█████     | 29/58 [00:02<00:01, 19.14it/s]Capturing num tokens (num_tokens=416 avail_mem=56.57 GB):  55%|█████▌    | 32/58 [00:02<00:01, 20.19it/s]Capturing num tokens (num_tokens=384 avail_mem=56.57 GB):  55%|█████▌    | 32/58 [00:02<00:01, 20.19it/s]Capturing num tokens (num_tokens=352 avail_mem=56.56 GB):  55%|█████▌    | 32/58 [00:02<00:01, 20.19it/s]Capturing num tokens (num_tokens=320 avail_mem=56.55 GB):  55%|█████▌    | 32/58 [00:02<00:01, 20.19it/s]Capturing num tokens (num_tokens=320 avail_mem=56.55 GB):  60%|██████    | 35/58 [00:02<00:01, 20.62it/s]Capturing num tokens (num_tokens=288 avail_mem=56.55 GB):  60%|██████    | 35/58 [00:02<00:01, 20.62it/s]

    Capturing num tokens (num_tokens=256 avail_mem=56.55 GB):  60%|██████    | 35/58 [00:02<00:01, 20.62it/s]Capturing num tokens (num_tokens=240 avail_mem=56.55 GB):  60%|██████    | 35/58 [00:02<00:01, 20.62it/s]Capturing num tokens (num_tokens=240 avail_mem=56.55 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.39it/s]Capturing num tokens (num_tokens=224 avail_mem=56.54 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.39it/s]Capturing num tokens (num_tokens=208 avail_mem=56.54 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.39it/s]Capturing num tokens (num_tokens=192 avail_mem=56.54 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.39it/s]

    Capturing num tokens (num_tokens=192 avail_mem=56.54 GB):  71%|███████   | 41/58 [00:02<00:00, 21.28it/s]Capturing num tokens (num_tokens=176 avail_mem=56.53 GB):  71%|███████   | 41/58 [00:02<00:00, 21.28it/s]Capturing num tokens (num_tokens=160 avail_mem=56.53 GB):  71%|███████   | 41/58 [00:02<00:00, 21.28it/s]Capturing num tokens (num_tokens=144 avail_mem=56.53 GB):  71%|███████   | 41/58 [00:02<00:00, 21.28it/s]Capturing num tokens (num_tokens=144 avail_mem=56.53 GB):  76%|███████▌  | 44/58 [00:02<00:00, 21.21it/s]Capturing num tokens (num_tokens=128 avail_mem=56.53 GB):  76%|███████▌  | 44/58 [00:02<00:00, 21.21it/s]Capturing num tokens (num_tokens=112 avail_mem=56.52 GB):  76%|███████▌  | 44/58 [00:02<00:00, 21.21it/s]

    Capturing num tokens (num_tokens=96 avail_mem=56.52 GB):  76%|███████▌  | 44/58 [00:02<00:00, 21.21it/s] Capturing num tokens (num_tokens=96 avail_mem=56.52 GB):  81%|████████  | 47/58 [00:02<00:00, 21.22it/s]Capturing num tokens (num_tokens=80 avail_mem=56.52 GB):  81%|████████  | 47/58 [00:02<00:00, 21.22it/s]Capturing num tokens (num_tokens=64 avail_mem=56.51 GB):  81%|████████  | 47/58 [00:02<00:00, 21.22it/s]Capturing num tokens (num_tokens=48 avail_mem=56.51 GB):  81%|████████  | 47/58 [00:02<00:00, 21.22it/s]Capturing num tokens (num_tokens=48 avail_mem=56.51 GB):  86%|████████▌ | 50/58 [00:02<00:00, 21.76it/s]Capturing num tokens (num_tokens=32 avail_mem=56.51 GB):  86%|████████▌ | 50/58 [00:02<00:00, 21.76it/s]

    Capturing num tokens (num_tokens=28 avail_mem=56.50 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.76it/s]Capturing num tokens (num_tokens=24 avail_mem=56.38 GB):  86%|████████▌ | 50/58 [00:03<00:00, 21.76it/s]Capturing num tokens (num_tokens=24 avail_mem=56.38 GB):  91%|█████████▏| 53/58 [00:03<00:00, 22.20it/s]Capturing num tokens (num_tokens=20 avail_mem=54.43 GB):  91%|█████████▏| 53/58 [00:03<00:00, 22.20it/s]Capturing num tokens (num_tokens=16 avail_mem=54.43 GB):  91%|█████████▏| 53/58 [00:03<00:00, 22.20it/s]Capturing num tokens (num_tokens=12 avail_mem=54.42 GB):  91%|█████████▏| 53/58 [00:03<00:00, 22.20it/s]

    Capturing num tokens (num_tokens=12 avail_mem=54.42 GB):  97%|█████████▋| 56/58 [00:03<00:00, 23.13it/s]Capturing num tokens (num_tokens=8 avail_mem=54.42 GB):  97%|█████████▋| 56/58 [00:03<00:00, 23.13it/s] Capturing num tokens (num_tokens=4 avail_mem=54.42 GB):  97%|█████████▋| 56/58 [00:03<00:00, 23.13it/s]Capturing num tokens (num_tokens=4 avail_mem=54.42 GB): 100%|██████████| 58/58 [00:03<00:00, 17.45it/s]


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
    Generated text:  Daryl and I am a professional graduate student. I recently graduated with my Bachelor of Science in Business Administration (B.S. in Business Administration) from the University of Wisconsin-Madison. I have been working as a financial advisor for 6 years now, having studied at the University of Wisconsin-Madison. I have a passion for helping people learn how to manage their money to achieve their goals. I have a Bachelor of Arts degree in Psychology from the University of Wisconsin-Madison and a minor in Psychology from the University of San Francisco. I am currently working in a new job and I am currently the Financial Advisor at WVS
    ===============================
    Prompt: The president of the United States is
    Generated text:  a country’s head of state and head of government. He or she is responsible for managing the country’s affairs and making decisions that affect the entire country. The president is the most powerful person in the United States government, and he or she has the authority to make decisions on matters that affect the nation. The president is responsible for appointing members of Congress and the Supreme Court, and he or she has the power to make decisions on federal laws.
    The president’s term of office is a maximum of four years. During that time, the president has the power to make decisions on all matters that affect the country. The president’s term of
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 正确
    B. 错误
    Answer:
    A
    
    Which of the following represents the correct order of the following sentences? ① An idea is like a secret. ② In this secret, there are many mountains and waters. ③ People cannot open all the secret's doors. ④ A secret is always something that cannot be disclosed. ⑤ The secret is that the mind cannot open all doors. 
    A. ①②⑤③④
    B. ①⑤②③④
    C. ②
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and we are excited to see it continue to evolve, transform, and create new possibilities. How is AI transforming the world today, and what are some of the major areas of AI development and innovation? In your opinion, what do you think is the most promising area of AI development, and how is it likely to impact the future of the world? AI is transforming the world in many ways. It has transformed the way we work, how we communicate, and how we learn. It is also transforming the way we live, from healthcare to transportation to entertainment.
    One of the most promising areas of AI development is in the field of robotics


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Name] with [Number] years of experience. I'm a [Number] year old, [Name] with [Number] years of experience. I'm a [Number] year old, [Name] with [Number] years of experience. I'm a [Number] year old, [Name] with [Number] years of experience. I'm a [Number] year old, [Name] with [Number] years of experience. I'm a [Number] year old, [Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many world-renowned landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife, fashion industry, and annual festivals such as the Eiffel Tower Parade. The city is a major economic and cultural center in Europe and plays a significant role in French politics and society. It is a popular tourist destination and is often referred to as the "City of Love" due to its romantic atmosphere and romantic architecture. Paris is a city of contrasts,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased use of AI in healthcare: AI is already being used in medical diagnosis and treatment, and it has the potential to revolutionize the field. AI-powered diagnostic tools could be used to identify diseases earlier and more accurately, potentially saving lives.
    
    2. AI in transportation: AI is already being used in self-driving cars, and it has the potential to revolutionize the transportation industry. AI-powered autonomous vehicles could reduce traffic congestion, improve safety, and decrease carbon emissions.
    
    3. AI in finance
    


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
    Generated text:  [Your Name] and I am a [insert your occupation] who specializes in [insert a skill or expertise that you have]. I have always been fascinated by the idea of [insert something interesting about yourself or your work that is not obvious from your job description, such as your drive for innovation, your passion for sustainability, or your commitment to environmental causes]. And I am always looking for opportunities to [insert what you are currently doing that you feel passionate about, such as teaching, helping people, writing, or any other activity that aligns with your interests]. If you have any questions or need assistance with [insert a relevant skill or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris, France, is the capital city of France. It is the largest city in the country and the seat of the French government and the largest city by area in Europe. Paris is known for its iconic landmarks, such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, and is home to many world-renowned museums, theaters, and food establishments. Its historical significance and cultural richness make it a popular tourist destination worldwide. Paris is also the second-largest city in the European Union in terms of population. The city has a rich cultural heritage and is a hub of art, architecture, and entertainment,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a combination of rapid development and consolidation, as well as the emergence of new technologies and innovations. Here are some possible future trends in AI:
    
    1. Increased integration with physical and environmental systems: AI is likely to become more integrated with physical and environmental systems such as transportation, healthcare, and urban planning. This will enable more efficient and effective use of resources and reduce the need for artificial intelligence to operate in isolation.
    
    2. Enhanced personalization and automation: AI will continue to improve the ability of machines to learn and make decisions based on the data it receives. Personalized recommendations and automation of tasks will become more common,


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

     [

    Age

    ].

     I

    ’m

     an

     [

    profession

    ]

     and

     I

     love

     [

    interest

     or

     hobby

    ]

     the

     most

    .

     I

     am

     [

    amb

    ition

     level

    ]

     in

     my

     career

    ,

     and

     I

     believe

     that

     my

     passion

     for

     [

    career

     field

    ]

     will

     help

     me

     advance

     my

     career

    .

     I

    ’m

     excited

     to

     get

     started

     and

     learn

     more

     about

     the

     [

    industry

     or

     field

    ]

     I

    ’m

     interested

     in

    .

     I

    ’m

     ready

     to

     make

     new

     connections

     and

     explore

     new

     ideas

     in

     my

     career

    .

     How

     can

     I

     reach

     out

     to

     you

     and

     what

     will

     you

     do

     to

     get

     to

     know

     me

    ?

     I

     am

     looking

     forward

     to

     our

     conversation

    .

     [

    Name

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     "

    City

     of

     Love

    ."


    That

    's

     correct

    !

     Paris

     is

     the

     "

    City

     of

     Love

    "

     in

     French

    ,

     and

     it

    's

     home

     to

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

     romantic

     atmosphere

     and

     numerous

     festivals

     throughout

     the

     year

    .

     Paris

     is

     a

     vibrant

     and

     fascinating

     city

     with

     a

     rich

     history

     and

     beautiful

     architecture

    .

     It

    's

     no

     wonder

     it

    's

     one

     of

     the

     most

     popular

     destinations

     in

     the

     world

    !

     (

    Source

    :

     Wikipedia

    )

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     combination

     of

     rapid

     progress

    ,

     innovation

    ,

     and

     disruptive

     changes

    .

     Here

     are

     some

     potential

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

     AI

     will

     continue

     to

     become

     more

     personalized

     and

     context

    -aware

    ,

     enabling

     more

     accurate

     and

     relevant

     predictions

     and

     recommendations

    .
    


    2

    .

     AI

     will

     become

     more

     capable

     of

     learning

     and

     adapting

     to

     new

     situations

    ,

     leading

     to

     a

     greater

     ability

     to

     handle

     complex

     and

     unpredictable

     situations

    .
    


    3

    .

     AI

     will

     continue

     to

     integrate

     with

     other

     technologies

    ,

     such

     as

     machine

     learning

     and

     deep

     learning

    ,

     to

     enable

     more

     powerful

     and

     efficient

     applications

    .
    


    4

    .

     AI

     will

     become

     more

     capable

     of

     performing

     tasks

     that

     are

     currently

     beyond

     the

     capabilities

     of

    



```python
llm.shutdown()
```
