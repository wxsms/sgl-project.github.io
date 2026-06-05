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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.71it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.71it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.71it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.71it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.73it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.73it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.73it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.73it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.14it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.14it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.14it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  7.14it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  7.14it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.93it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.93it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.93it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.93it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.93it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.75it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.75it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 22.66it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 22.66it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 22.66it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 22.66it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 22.66it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 22.66it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 27.70it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 30.03it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 30.03it/s]

    Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 30.03it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 30.03it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 30.03it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 30.03it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 34.26it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 34.26it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 34.26it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 34.26it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 34.26it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 35.36it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 35.36it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 35.36it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 35.36it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 35.36it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 35.36it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 37.99it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 37.99it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 37.99it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 37.99it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 37.99it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 37.99it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 41.02it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 41.02it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.59 GB):   2%|▏         | 1/58 [00:00<00:07,  8.09it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.57 GB):   2%|▏         | 1/58 [00:00<00:07,  8.09it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.57 GB):   3%|▎         | 2/58 [00:00<00:06,  9.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.57 GB):   3%|▎         | 2/58 [00:00<00:06,  9.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.56 GB):   3%|▎         | 2/58 [00:00<00:06,  9.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.56 GB):   7%|▋         | 4/58 [00:00<00:04, 10.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.56 GB):   7%|▋         | 4/58 [00:00<00:04, 10.92it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.55 GB):   7%|▋         | 4/58 [00:00<00:04, 10.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.55 GB):  10%|█         | 6/58 [00:00<00:04, 12.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.53 GB):  10%|█         | 6/58 [00:00<00:04, 12.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.52 GB):  10%|█         | 6/58 [00:00<00:04, 12.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.52 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.52 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.13it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=53.51 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.50 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.50 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.50 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.49 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.49 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.49 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.49 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.49 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.74it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=53.48 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.48 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.48 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.48 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.48 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.46 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.76it/s]Capturing num tokens (num_tokens=960 avail_mem=53.47 GB):  34%|███▍      | 20/58 [00:01<00:01, 29.76it/s] Capturing num tokens (num_tokens=896 avail_mem=53.47 GB):  34%|███▍      | 20/58 [00:01<00:01, 29.76it/s]Capturing num tokens (num_tokens=832 avail_mem=53.46 GB):  34%|███▍      | 20/58 [00:01<00:01, 29.76it/s]Capturing num tokens (num_tokens=768 avail_mem=53.46 GB):  34%|███▍      | 20/58 [00:01<00:01, 29.76it/s]Capturing num tokens (num_tokens=768 avail_mem=53.46 GB):  43%|████▎     | 25/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=704 avail_mem=53.46 GB):  43%|████▎     | 25/58 [00:01<00:00, 34.93it/s]

    Capturing num tokens (num_tokens=640 avail_mem=53.45 GB):  43%|████▎     | 25/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=576 avail_mem=53.45 GB):  43%|████▎     | 25/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=512 avail_mem=53.44 GB):  43%|████▎     | 25/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=480 avail_mem=53.45 GB):  43%|████▎     | 25/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=480 avail_mem=53.45 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=448 avail_mem=53.45 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=416 avail_mem=53.45 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=384 avail_mem=53.45 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=352 avail_mem=53.44 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=320 avail_mem=53.44 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=320 avail_mem=53.44 GB):  60%|██████    | 35/58 [00:01<00:00, 40.88it/s]Capturing num tokens (num_tokens=288 avail_mem=52.30 GB):  60%|██████    | 35/58 [00:01<00:00, 40.88it/s]

    Capturing num tokens (num_tokens=256 avail_mem=52.30 GB):  60%|██████    | 35/58 [00:01<00:00, 40.88it/s]Capturing num tokens (num_tokens=240 avail_mem=52.29 GB):  60%|██████    | 35/58 [00:01<00:00, 40.88it/s]Capturing num tokens (num_tokens=224 avail_mem=52.29 GB):  60%|██████    | 35/58 [00:01<00:00, 40.88it/s]

    Capturing num tokens (num_tokens=208 avail_mem=53.39 GB):  60%|██████    | 35/58 [00:01<00:00, 40.88it/s]Capturing num tokens (num_tokens=208 avail_mem=53.39 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.72it/s]Capturing num tokens (num_tokens=192 avail_mem=53.39 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.72it/s]Capturing num tokens (num_tokens=176 avail_mem=52.40 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.72it/s]

    Capturing num tokens (num_tokens=160 avail_mem=52.40 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.72it/s]Capturing num tokens (num_tokens=144 avail_mem=52.39 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.72it/s]Capturing num tokens (num_tokens=144 avail_mem=52.39 GB):  76%|███████▌  | 44/58 [00:01<00:00, 20.37it/s]Capturing num tokens (num_tokens=128 avail_mem=53.37 GB):  76%|███████▌  | 44/58 [00:01<00:00, 20.37it/s]

    Capturing num tokens (num_tokens=112 avail_mem=52.45 GB):  76%|███████▌  | 44/58 [00:02<00:00, 20.37it/s]Capturing num tokens (num_tokens=96 avail_mem=52.45 GB):  76%|███████▌  | 44/58 [00:02<00:00, 20.37it/s] Capturing num tokens (num_tokens=96 avail_mem=52.45 GB):  81%|████████  | 47/58 [00:02<00:00, 18.01it/s]Capturing num tokens (num_tokens=80 avail_mem=53.37 GB):  81%|████████  | 47/58 [00:02<00:00, 18.01it/s]

    Capturing num tokens (num_tokens=64 avail_mem=52.50 GB):  81%|████████  | 47/58 [00:02<00:00, 18.01it/s]Capturing num tokens (num_tokens=48 avail_mem=52.50 GB):  81%|████████  | 47/58 [00:02<00:00, 18.01it/s]Capturing num tokens (num_tokens=48 avail_mem=52.50 GB):  86%|████████▌ | 50/58 [00:02<00:00, 16.26it/s]Capturing num tokens (num_tokens=32 avail_mem=53.36 GB):  86%|████████▌ | 50/58 [00:02<00:00, 16.26it/s]

    Capturing num tokens (num_tokens=28 avail_mem=53.35 GB):  86%|████████▌ | 50/58 [00:02<00:00, 16.26it/s]Capturing num tokens (num_tokens=28 avail_mem=53.35 GB):  90%|████████▉ | 52/58 [00:02<00:00, 16.40it/s]Capturing num tokens (num_tokens=24 avail_mem=52.56 GB):  90%|████████▉ | 52/58 [00:02<00:00, 16.40it/s]Capturing num tokens (num_tokens=20 avail_mem=52.56 GB):  90%|████████▉ | 52/58 [00:02<00:00, 16.40it/s]

    Capturing num tokens (num_tokens=20 avail_mem=52.56 GB):  93%|█████████▎| 54/58 [00:02<00:00, 15.42it/s]Capturing num tokens (num_tokens=16 avail_mem=53.35 GB):  93%|█████████▎| 54/58 [00:02<00:00, 15.42it/s]Capturing num tokens (num_tokens=12 avail_mem=52.61 GB):  93%|█████████▎| 54/58 [00:02<00:00, 15.42it/s]Capturing num tokens (num_tokens=12 avail_mem=52.61 GB):  97%|█████████▋| 56/58 [00:02<00:00, 15.10it/s]Capturing num tokens (num_tokens=8 avail_mem=52.61 GB):  97%|█████████▋| 56/58 [00:02<00:00, 15.10it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=53.33 GB):  97%|█████████▋| 56/58 [00:02<00:00, 15.10it/s]Capturing num tokens (num_tokens=4 avail_mem=53.33 GB): 100%|██████████| 58/58 [00:02<00:00, 14.95it/s]Capturing num tokens (num_tokens=4 avail_mem=53.33 GB): 100%|██████████| 58/58 [00:02<00:00, 19.46it/s]


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
    Generated text:  Eliza, a software developer specializing in AI-driven marketing. I am passionate about using machine learning to help businesses create more effective marketing campaigns. I have a strong background in Python and R, and have worked with a variety of programming languages and frameworks to develop and deploy my software. I have a keen eye for detail and a strong attention to detail, which I apply when developing my software. I am always looking for ways to improve my skills and stay up to date with the latest advancements in AI and machine learning. Thank you for considering me for a role. How can I get started with a project that involves using machine learning for a marketing
    ===============================
    Prompt: The president of the United States is
    Generated text:  to be selected from the members of the Communist Party of China. The president of the United States is to be selected from the members of the Communist Party of China. This is a _______.
    A: Compound event
    B: Random event
    C: Certain event
    D: Impossible event
    To determine the nature of the event where the president of the United States is to be selected from the members of the Communist Party of China, let's analyze the situation step by step.
    
    1. **Identify the event**: We need to select the president of the United States from the members of the Communist Party of China.
    2. **Understanding
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. London
    B. Paris
    C. Brussels
    D. Madrid
    Answer:
    
    B
    
    A 40-year-old male patient has been experiencing recurrent coughing and hemoptysis over the past 6 months, with hemoptysis occurring 5 to 7 times per month. Which of the following conditions is most likely to be the cause of the patient's hemoptysis?
    A. Right-sided pneumothorax
    B. Tuberculosis
    C. Pulmonary embolism
    D. Lung cancer
    E. Bronchiectasis
    Answer:
    
    B
    
    Given the function $ f(x) = \sin
    ===============================
    Prompt: The future of AI is
    Generated text:  very promising. But it's also very complex. In this blog post I'll discuss some of the issues we face when building AI systems and the best practices to handle them. We'll also discuss the importance of data privacy and ethics when using AI.
    
    Data privacy and ethics
    
    The use of AI systems requires a great deal of data. But these data can also be collected and used by the system without the users' consent. This can have serious consequences, including the misuse of data and privacy violations.
    
    The ethical implications of AI also need to be considered. For example, if an AI system is designed to make decisions that are harmful or unethical


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character here, such as "funny, adventurous, and always looking for new experiences."]. I'm always looking for new adventures and challenges, and I'm always eager to learn and grow. What's your favorite hobby or activity? I'm always looking for new ways to challenge myself and expand my horizons. What's your favorite book or movie? I'm always drawn to stories that have a unique twist
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and the largest city in the European Union. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is also a major transportation hub, with the Eiffel Tower serving as a symbol of the city's importance in international trade. Paris is a popular tourist destination, known for its fashion, art, and cuisine. It is also home to many famous museums, including the Louvre and the Mus
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions between humans and machines.
    
    2. Greater reliance on machine learning: Machine learning will continue to play a larger role in AI, with more sophisticated algorithms and models being developed to improve accuracy and efficiency.
    
    3. Increased focus on ethical considerations: As AI becomes more integrated with human intelligence, there will be increased focus on ethical considerations, including issues such as bias, transparency, and accountability.
    
    4. Greater use of AI in healthcare: AI is
    


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
    Generated text:  [Name] and I am a [type of character]. I am [age] years old and [occupation]. I am a [job title] with a [interest or passion] that has always been my [value].
    
    What do you do for a living? What's something you do on a regular basis? Can you tell us about your favorite hobby or activity?
    (Repeat the above sentence, replacing [Name], [type of character], [age], [occupation], [job title], [interest or passion], [value], [regular basis], and [favorite hobby/act]). Hello, my name is [Name] and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement encapsulates the basic facts about the capital city of France, including its name, country, and the fact that it is the capital. It is a concise and informative response that provides a clear understanding of the city's location, its role in French politics, and its significance to France and the world. 
    
    Alternative statements that may be relevant include: 
    - "The capital of France is Paris."
    - "Paris is the capital of France."
    - "France has Paris as its capital." 
    
    However, none of these alternatives directly address the concept of the "capital city" as it relates to the French government and administration
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly unpredictable and will continue to evolve rapidly. Here are some potential trends that may be seen in the development of artificial intelligence in the coming years:
    
    1. Increased accuracy: As technology improves, AI systems may become more accurate in their predictions and decisions. For example, AI may be able to predict the weather, healthcare outcomes, or the likelihood of natural disasters with greater accuracy than humans.
    
    2. Autonomous decision-making: AI is already capable of making decisions without human intervention, but in the future, it may become more autonomous. This could lead to a more efficient and cost-effective way of making decisions.
    
    3. Enhanced emotional intelligence: AI may


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

    ’m

     a

     

    2

    7

    -year

    -old

     software

     engineer

     with

     a

     passion

     for

     innovation

     and

     solving

     complex

     problems

    .

     I

     enjoy

     working

     with

     teams

    ,

     learning

     new

     technologies

    ,

     and

     experimenting

     with

     new

     ideas

    .

     My

     favorite

     hobbies

     include

     reading

    ,

     listening

     to

     music

    ,

     and

     spending

     time

     outdoors

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

     more

     about

     the

     world

     around

     me

    .

     Thank

     you

     for

     the

     opportunity

     to

     meet

     you

    !

     

    🌟

    ✨

    


    Hi

     there

    !

     I

    'm

     [

    Your

     Name

    ]

     from

     [

    Your

     background

    ],

     and

     I

    've

     been

     working

     in

     tech

     for

     

    3

     years

     now

    .

     My

     passion

     for

     coding

     and

     problem

    -solving

     has

     always

     driven

     me

     forward

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     in

     the

     country

    .

     It

     is

     known

     for

     its

     towering

     architecture

    ,

     vibrant

     streets

    ,

     and

     romantic

     ambiance

    .

     The

     city

     is

     home

     to

     several

     landmarks

     such

     as

     Notre

    -D

    ame

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

    .

     Paris

     is

     also

     a

     cultural

     hub

     and

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     other

     cultural

     institutions

    .

     It

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     has

     a

     rich

     history

     dating

     back

     over

     

    2

    ,

    5

    0

    0

     years

    .

     The

     city

     is

     known

     for

     its

     fine

     cuisine

    ,

     art

    ,

     and

     entertainment

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     Paris

     has

     a

     strong

     focus

     on

     urban

     design

     and

     a

     commitment

     to

     preserving

     its

     historical

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     range

     of

     trends

     that

     are

     shaping

     the

     landscape

     of

     the

     field

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

     Personal

    ized

     AI

    :

     Personal

    ized

     AI

     will

     continue

     to

     evolve

    ,

     allowing

     machines

     to

     learn

     from

     user

     data

     and

     provide

     more

     accurate

     and

     relevant

     recommendations

    .

     This

     will

     be

     particularly

     important

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     customer

     service

    .
    


    2

    .

     Eth

    ical

     AI

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     increased

     scrutiny

     of

     its

     ethical

     implications

    .

     This

     will

     lead

     to

     more

     regulations

     and

     guidelines

     to

     ensure

     that

     AI

     is

     used

     responsibly

     and

     in

     a

     way

     that

     benefits

     society

     as

     a

     whole

    .
    


    3

    .

     Autonomous

     AI

    :

    



```python
llm.shutdown()
```
