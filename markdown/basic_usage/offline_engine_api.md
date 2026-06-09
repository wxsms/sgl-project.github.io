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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:03,  1.15s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.69it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.69it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.69it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.76it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.76it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.76it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.76it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:09,  4.80it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.26it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.26it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.26it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  7.26it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  7.26it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 13.86it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 13.86it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 13.86it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 13.86it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 15.81it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 15.81it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 15.81it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 15.81it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 17.71it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 17.71it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 17.71it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 17.71it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 18.79it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 18.79it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 18.79it/s]

    Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 18.79it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 20.40it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 20.40it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 20.40it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 20.40it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 22.24it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 22.24it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 22.24it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 22.24it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.59it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.59it/s]

    Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.59it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.59it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 25.04it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 25.04it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 25.04it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 25.04it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 25.31it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 25.31it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 25.31it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 25.31it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:06<00:00, 25.63it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:06<00:00, 25.63it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:06<00:00, 25.63it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:06<00:00, 25.63it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:06<00:00, 25.63it/s]

    Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 27.45it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 27.45it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 27.45it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:06<00:00, 27.45it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 28.01it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 28.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=39.04 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=39.04 GB):   2%|▏         | 1/58 [00:00<00:13,  4.33it/s]Capturing num tokens (num_tokens=7680 avail_mem=39.00 GB):   2%|▏         | 1/58 [00:00<00:13,  4.33it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=39.00 GB):   3%|▎         | 2/58 [00:00<00:14,  3.98it/s]Capturing num tokens (num_tokens=7168 avail_mem=39.00 GB):   3%|▎         | 2/58 [00:00<00:14,  3.98it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=39.00 GB):   5%|▌         | 3/58 [00:00<00:12,  4.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=39.00 GB):   5%|▌         | 3/58 [00:00<00:12,  4.30it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=39.00 GB):   7%|▋         | 4/58 [00:00<00:12,  4.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=39.00 GB):   7%|▋         | 4/58 [00:00<00:12,  4.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=39.00 GB):   9%|▊         | 5/58 [00:01<00:11,  4.45it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.99 GB):   9%|▊         | 5/58 [00:01<00:11,  4.45it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=38.99 GB):  10%|█         | 6/58 [00:01<00:10,  4.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.98 GB):  10%|█         | 6/58 [00:01<00:10,  4.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.98 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.98 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.87it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=38.98 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.98 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.98 GB):  16%|█▌        | 9/58 [00:01<00:09,  5.28it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.98 GB):  16%|█▌        | 9/58 [00:01<00:09,  5.28it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=38.98 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.97 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.97 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.97 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.65it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=38.97 GB):  21%|██        | 12/58 [00:02<00:07,  5.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.96 GB):  21%|██        | 12/58 [00:02<00:07,  5.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.96 GB):  22%|██▏       | 13/58 [00:02<00:07,  5.86it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.96 GB):  22%|██▏       | 13/58 [00:02<00:07,  5.86it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=38.96 GB):  22%|██▏       | 13/58 [00:02<00:07,  5.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.96 GB):  26%|██▌       | 15/58 [00:02<00:05,  7.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.96 GB):  26%|██▌       | 15/58 [00:02<00:05,  7.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.95 GB):  26%|██▌       | 15/58 [00:02<00:05,  7.64it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=38.95 GB):  29%|██▉       | 17/58 [00:02<00:04,  8.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.95 GB):  29%|██▉       | 17/58 [00:02<00:04,  8.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.95 GB):  29%|██▉       | 17/58 [00:02<00:04,  8.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.95 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.94 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.92 GB):  33%|███▎      | 19/58 [00:03<00:03, 10.44it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=38.92 GB):  36%|███▌      | 21/58 [00:03<00:03, 12.04it/s]Capturing num tokens (num_tokens=960 avail_mem=38.94 GB):  36%|███▌      | 21/58 [00:03<00:03, 12.04it/s] Capturing num tokens (num_tokens=896 avail_mem=38.94 GB):  36%|███▌      | 21/58 [00:03<00:03, 12.04it/s]Capturing num tokens (num_tokens=896 avail_mem=38.94 GB):  40%|███▉      | 23/58 [00:03<00:02, 13.38it/s]Capturing num tokens (num_tokens=832 avail_mem=38.93 GB):  40%|███▉      | 23/58 [00:03<00:02, 13.38it/s]

    Capturing num tokens (num_tokens=768 avail_mem=38.93 GB):  40%|███▉      | 23/58 [00:03<00:02, 13.38it/s]Capturing num tokens (num_tokens=768 avail_mem=38.93 GB):  43%|████▎     | 25/58 [00:03<00:02, 11.13it/s]Capturing num tokens (num_tokens=704 avail_mem=38.93 GB):  43%|████▎     | 25/58 [00:03<00:02, 11.13it/s]

    Capturing num tokens (num_tokens=640 avail_mem=38.92 GB):  43%|████▎     | 25/58 [00:03<00:02, 11.13it/s]Capturing num tokens (num_tokens=640 avail_mem=38.92 GB):  47%|████▋     | 27/58 [00:03<00:03, 10.19it/s]Capturing num tokens (num_tokens=576 avail_mem=38.92 GB):  47%|████▋     | 27/58 [00:03<00:03, 10.19it/s]

    Capturing num tokens (num_tokens=512 avail_mem=38.91 GB):  47%|████▋     | 27/58 [00:03<00:03, 10.19it/s]Capturing num tokens (num_tokens=512 avail_mem=38.91 GB):  50%|█████     | 29/58 [00:03<00:02,  9.82it/s]Capturing num tokens (num_tokens=480 avail_mem=38.92 GB):  50%|█████     | 29/58 [00:03<00:02,  9.82it/s]

    Capturing num tokens (num_tokens=448 avail_mem=38.92 GB):  50%|█████     | 29/58 [00:04<00:02,  9.82it/s]Capturing num tokens (num_tokens=448 avail_mem=38.92 GB):  53%|█████▎    | 31/58 [00:04<00:02,  9.74it/s]Capturing num tokens (num_tokens=416 avail_mem=38.92 GB):  53%|█████▎    | 31/58 [00:04<00:02,  9.74it/s]

    Capturing num tokens (num_tokens=384 avail_mem=38.91 GB):  53%|█████▎    | 31/58 [00:04<00:02,  9.74it/s]Capturing num tokens (num_tokens=384 avail_mem=38.91 GB):  57%|█████▋    | 33/58 [00:04<00:02,  9.74it/s]Capturing num tokens (num_tokens=352 avail_mem=38.90 GB):  57%|█████▋    | 33/58 [00:04<00:02,  9.74it/s]

    Capturing num tokens (num_tokens=320 avail_mem=38.90 GB):  57%|█████▋    | 33/58 [00:04<00:02,  9.74it/s]Capturing num tokens (num_tokens=320 avail_mem=38.90 GB):  60%|██████    | 35/58 [00:04<00:02,  9.61it/s]Capturing num tokens (num_tokens=288 avail_mem=38.90 GB):  60%|██████    | 35/58 [00:04<00:02,  9.61it/s]Capturing num tokens (num_tokens=256 avail_mem=38.90 GB):  60%|██████    | 35/58 [00:04<00:02,  9.61it/s]

    Capturing num tokens (num_tokens=256 avail_mem=38.90 GB):  64%|██████▍   | 37/58 [00:04<00:01, 10.99it/s]Capturing num tokens (num_tokens=240 avail_mem=38.89 GB):  64%|██████▍   | 37/58 [00:04<00:01, 10.99it/s]Capturing num tokens (num_tokens=224 avail_mem=38.89 GB):  64%|██████▍   | 37/58 [00:04<00:01, 10.99it/s]Capturing num tokens (num_tokens=208 avail_mem=38.88 GB):  64%|██████▍   | 37/58 [00:04<00:01, 10.99it/s]Capturing num tokens (num_tokens=208 avail_mem=38.88 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.81it/s]Capturing num tokens (num_tokens=192 avail_mem=38.88 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.81it/s]

    Capturing num tokens (num_tokens=176 avail_mem=38.88 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.81it/s]Capturing num tokens (num_tokens=160 avail_mem=38.88 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.81it/s]Capturing num tokens (num_tokens=160 avail_mem=38.88 GB):  74%|███████▍  | 43/58 [00:05<00:01, 14.05it/s]Capturing num tokens (num_tokens=144 avail_mem=37.90 GB):  74%|███████▍  | 43/58 [00:05<00:01, 14.05it/s]

    Capturing num tokens (num_tokens=128 avail_mem=37.90 GB):  74%|███████▍  | 43/58 [00:05<00:01, 14.05it/s]Capturing num tokens (num_tokens=128 avail_mem=37.90 GB):  78%|███████▊  | 45/58 [00:05<00:01, 10.88it/s]Capturing num tokens (num_tokens=112 avail_mem=37.90 GB):  78%|███████▊  | 45/58 [00:05<00:01, 10.88it/s]

    Capturing num tokens (num_tokens=96 avail_mem=37.90 GB):  78%|███████▊  | 45/58 [00:05<00:01, 10.88it/s] Capturing num tokens (num_tokens=96 avail_mem=37.90 GB):  81%|████████  | 47/58 [00:05<00:01,  9.47it/s]Capturing num tokens (num_tokens=80 avail_mem=38.83 GB):  81%|████████  | 47/58 [00:05<00:01,  9.47it/s]

    Capturing num tokens (num_tokens=64 avail_mem=38.83 GB):  81%|████████  | 47/58 [00:05<00:01,  9.47it/s]Capturing num tokens (num_tokens=64 avail_mem=38.83 GB):  84%|████████▍ | 49/58 [00:05<00:01,  8.79it/s]Capturing num tokens (num_tokens=48 avail_mem=38.00 GB):  84%|████████▍ | 49/58 [00:05<00:01,  8.79it/s]

    Capturing num tokens (num_tokens=48 avail_mem=38.00 GB):  86%|████████▌ | 50/58 [00:06<00:00,  8.30it/s]Capturing num tokens (num_tokens=32 avail_mem=38.00 GB):  86%|████████▌ | 50/58 [00:06<00:00,  8.30it/s]Capturing num tokens (num_tokens=32 avail_mem=38.00 GB):  88%|████████▊ | 51/58 [00:06<00:00,  7.87it/s]Capturing num tokens (num_tokens=28 avail_mem=37.99 GB):  88%|████████▊ | 51/58 [00:06<00:00,  7.87it/s]

    Capturing num tokens (num_tokens=28 avail_mem=37.99 GB):  90%|████████▉ | 52/58 [00:06<00:00,  7.78it/s]Capturing num tokens (num_tokens=24 avail_mem=38.82 GB):  90%|████████▉ | 52/58 [00:06<00:00,  7.78it/s]Capturing num tokens (num_tokens=24 avail_mem=38.82 GB):  91%|█████████▏| 53/58 [00:06<00:00,  7.83it/s]Capturing num tokens (num_tokens=20 avail_mem=38.81 GB):  91%|█████████▏| 53/58 [00:06<00:00,  7.83it/s]

    Capturing num tokens (num_tokens=20 avail_mem=38.81 GB):  93%|█████████▎| 54/58 [00:06<00:00,  7.97it/s]Capturing num tokens (num_tokens=16 avail_mem=38.04 GB):  93%|█████████▎| 54/58 [00:06<00:00,  7.97it/s]Capturing num tokens (num_tokens=16 avail_mem=38.04 GB):  95%|█████████▍| 55/58 [00:06<00:00,  7.52it/s]Capturing num tokens (num_tokens=12 avail_mem=38.04 GB):  95%|█████████▍| 55/58 [00:06<00:00,  7.52it/s]

    Capturing num tokens (num_tokens=12 avail_mem=38.04 GB):  97%|█████████▋| 56/58 [00:06<00:00,  7.27it/s]Capturing num tokens (num_tokens=8 avail_mem=38.04 GB):  97%|█████████▋| 56/58 [00:06<00:00,  7.27it/s] Capturing num tokens (num_tokens=8 avail_mem=38.04 GB):  98%|█████████▊| 57/58 [00:07<00:00,  7.26it/s]Capturing num tokens (num_tokens=4 avail_mem=38.80 GB):  98%|█████████▊| 57/58 [00:07<00:00,  7.26it/s]

    Capturing num tokens (num_tokens=4 avail_mem=38.80 GB): 100%|██████████| 58/58 [00:07<00:00,  7.22it/s]Capturing num tokens (num_tokens=4 avail_mem=38.80 GB): 100%|██████████| 58/58 [00:07<00:00,  8.08it/s]


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
    Generated text:  Grace. I am an American actress and I have appeared in a lot of different television and film roles. I am also a volunteer for the United Nations Relief and Works Agency for Palestine Refugees, a non-profit organization. I recently took part in a workshop about "Human Rights and Peace". The event was hosted by my friend and I also worked in the organization. I met many people from different countries and religions. We had a fun time, and we discussed many topics. Our guide was very friendly and helpful, and we were all very comfortable. It was a very memorable event. What can you tell us about the Human Rights and Peace
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking official who holds a prominent position in the government, often serving as a prime minister or deputy president of the United States. The term of office for the president is five years. The president is responsible for the day-to-day functioning of the government and for making decisions that impact the country's policies and laws. They are also responsible for appointing and confirming important federal officers, such as the Secretary of State, the Secretary of the Treasury, the Attorney General, and the Secretary of Homeland Security. The president is the head of the executive branch, and their role is to represent the country and ensure that the laws are implemented and that
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Rome
    C. Berlin
    D. Moscow
    Answer:
    A
    
    Which of the following is a correct evaluation of the project's main tasks? ①Objective ②Purpose ③Time ④Budget ⑤Details ⑥Quality
    A. ①②③④⑤
    B. ①②④⑤⑥
    C. ①③④⑤⑥
    D. ①②③④⑤⑥
    Answer:
    D
    
    The most likely type
    ===============================
    Prompt: The future of AI is
    Generated text:  now
    
    In this video, Christian Szeliski discusses how AI is taking shape, the implications of the data avalanche, and the changing landscape for AI research.
    
    Christian Szeliski is a professor in the Department of Computer Science at Stanford University and the director of the CSAIL Research Group at MIT. He has published over 160 papers and is known for his work with computer vision, AI, and machine learning. His new book, The New AI, was published in March.
    
    Christian Szeliski
    
    Christian Szeliski: Hi, everyone. Welcome to the third video of the course, The New AI


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. And what's your favorite hobby or activity? I'm always looking for new experiences and adventures, so I'm always up for trying new things. What's your favorite book or movie? I'm always on the lookout for new reads and movies to enjoy. And what's your favorite color? I'm always looking for a new way to express myself and express myself in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a cultural and economic center with a rich history dating back to the Middle Ages. The city is home to many famous museums, including the Louvre and the Musée d'Orsay, and is a major hub for international trade and diplomacy. Paris is a popular tourist destination and a major center for French culture and cuisine. The city is also known for its fashion industry, with many famous designers and boutiques. Overall, Paris is a vibrant and dynamic city with a rich history and a strong sense
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI systems that can better understand and respond to human needs.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes
    


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
    Generated text:  [Name]. I am an [age] year old [gender] [occupation]. I have always been [a short, memorable quote or proverb] about myself, and it has stuck with me since childhood. I am always looking for ways to improve myself and reach my full potential. My goal is to make the world a better place by using my skills and knowledge. I believe that hard work, dedication, and perseverance will lead me to success, and that I am ready to take on any challenge that comes my way. Thank you for considering my humble introduction. #SelfIntroduction #Personality #PositiveAttitude
    
    Hey there!
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and the capital of the country. It is renowned for its art, cuisine, and fashion, and it is the world's largest city in terms of population. It is home to iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its cultural diversity and its role as the home of the French Revolution and its current popularity as a tourist destination. Its prominence on the global stage makes it a UNESCO World Heritage Site. Paris is often referred to as the "City of Light" and its wealth of historical landmarks and cultural activities
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and dynamic, and it is impossible to predict with certainty what will be the next big innovation or development in the field. However, here are some possible trends that could potentially influence the future of AI:
    
    1. Increased focus on ethical and responsible AI: As more AI systems are used for critical tasks such as healthcare, transportation, and security, there will likely be increased pressure to ensure that AI systems are developed and used ethically and responsibly. This could lead to increased focus on ethical principles such as transparency, accountability, and fairness in the development and use of AI.
    
    2. AI will become more integrated into our daily lives: As


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

    Age

    ]

     year

     old

    .

     I

    'm

     currently

     [

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     My

     favorite

     hobby

     is

     [

    Favorite

     Hobby

    /

    Activity

    ].

     What

     do

     you

     think

     I

     look

     like

     and

     what

     do

     I

     look

     like

    ?

     I

    'm

     tall

    ,

     [

    Height

    ]

     and

     [

    Weight

    ].

     I

     have

     [

    Hair

     Color

    ]

     and

     [

    Eye

     Color

    ].

     I

     have

     [

    Fitness

     Level

    ]

     and

     am

     [

    Physical

     Activity

     Level

    ].

     I

    'm

     [

    Gender

    ].

     I

     have

     a

     [

    Physical

     Feature

    ]

     and

     [

    Other

     Physical

     Feature

    ].

     I

     have

     [

    Other

     Physical

     Feature

    ]

     and

     [

    Other

     Physical

     Feature

    ].

     How

     would

     you

     describe

     yourself

    ?


    As

     an

     AI

     language

     model

    ,

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     north

    western

     part

     of

     the

     country

     and

     is

     the

     largest

     city

     in

     terms

     of

     population

    ,

     with

     a

     population

     of

     approximately

     

    2

    .

    7

     million

     people

    .

     Paris

     is

     also

     the

     seat

     of

     government

    ,

     the

     French

     parliament

    ,

     and

     the

     French

     government

    ,

     and

     is

     a

     significant

     cultural

     and

     historical

     center

    .

     It

     is

     also

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

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     known

     for

     its

     beautiful

     architecture

    ,

     museums

    ,

     and

     food

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     attracts

     millions

     of

     visitors

     each

     year

    .

     It

     is

     known

     for

     its

     historical

     sites

    ,

     fashion

    ,

     and

     entertainment

    .

     The

     city

     is

     also

     famous

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     trends

     that

     are

     expected

     to

     continue

     and

     evolve

     in

     the

     coming

     years

    .

     Some

     of

     the

     key

     trends

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

     and

     medicine

    :

     AI

     is

     already

     being

     used

     to

     improve

     patient

     care

     and

     treatment

     outcomes

    ,

     and

     we

     can

     expect

     to

     see

     even

     more

     innovative

     uses

     in

     the

     coming

     years

    .
    


    2

    .

     More

     widespread

     adoption

     of

     AI

     in

     industry

    :

     As

     AI

     becomes

     more

     sophisticated

     and

     accessible

     to

     the

     public

    ,

     we

     can

     expect

     to

     see

     more

     widespread

     adoption

     in

     industries

     such

     as

     manufacturing

    ,

     transportation

    ,

     and

     finance

    .
    


    3

    .

     Greater

     emphasis

     on

     privacy

     and

     security

    :

     With

     the

     increasing

     amount

     of

     data

     being

     collected

     and

     processed

     by

     AI

    



```python
llm.shutdown()
```
