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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.98it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.67it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.85it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.85it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.59it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.59it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.59it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.59it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.59it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.59it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.59it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.59it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.59it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 36.33it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 36.33it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 36.33it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 36.33it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.71 GB):   2%|▏         | 1/58 [00:00<00:07,  7.33it/s]Capturing num tokens (num_tokens=7680 avail_mem=52.67 GB):   2%|▏         | 1/58 [00:00<00:07,  7.33it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=52.67 GB):   3%|▎         | 2/58 [00:00<00:07,  7.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.67 GB):   3%|▎         | 2/58 [00:00<00:07,  7.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.67 GB):   5%|▌         | 3/58 [00:00<00:07,  7.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.67 GB):   5%|▌         | 3/58 [00:00<00:07,  7.48it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=52.67 GB):   7%|▋         | 4/58 [00:00<00:07,  7.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.67 GB):   7%|▋         | 4/58 [00:00<00:07,  7.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.67 GB):   9%|▊         | 5/58 [00:00<00:06,  7.94it/s]Capturing num tokens (num_tokens=5632 avail_mem=52.66 GB):   9%|▊         | 5/58 [00:00<00:06,  7.94it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=52.66 GB):  10%|█         | 6/58 [00:00<00:06,  8.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.65 GB):  10%|█         | 6/58 [00:00<00:06,  8.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.65 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.65 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.75it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.65 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.75it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=52.65 GB):  16%|█▌        | 9/58 [00:01<00:04,  9.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.65 GB):  16%|█▌        | 9/58 [00:01<00:04,  9.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.64 GB):  16%|█▌        | 9/58 [00:01<00:04,  9.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.64 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=52.64 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.88it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=52.63 GB):  19%|█▉        | 11/58 [00:01<00:04, 10.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.63 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.63 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.63 GB):  22%|██▏       | 13/58 [00:01<00:03, 11.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.63 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.63 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.03it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=52.62 GB):  26%|██▌       | 15/58 [00:01<00:03, 13.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.62 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.62 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.62 GB):  29%|██▉       | 17/58 [00:01<00:02, 14.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.62 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=52.61 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.54it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=52.59 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.54it/s]Capturing num tokens (num_tokens=960 avail_mem=52.61 GB):  33%|███▎      | 19/58 [00:01<00:02, 15.54it/s] Capturing num tokens (num_tokens=960 avail_mem=52.61 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.30it/s]Capturing num tokens (num_tokens=896 avail_mem=52.61 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.30it/s]Capturing num tokens (num_tokens=832 avail_mem=52.60 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.30it/s]Capturing num tokens (num_tokens=768 avail_mem=52.60 GB):  38%|███▊      | 22/58 [00:01<00:02, 17.30it/s]

    Capturing num tokens (num_tokens=768 avail_mem=52.60 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.02it/s]Capturing num tokens (num_tokens=704 avail_mem=52.60 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.02it/s]Capturing num tokens (num_tokens=640 avail_mem=52.59 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.02it/s]Capturing num tokens (num_tokens=576 avail_mem=52.59 GB):  43%|████▎     | 25/58 [00:02<00:01, 19.02it/s]Capturing num tokens (num_tokens=576 avail_mem=52.59 GB):  48%|████▊     | 28/58 [00:02<00:01, 20.20it/s]Capturing num tokens (num_tokens=512 avail_mem=52.58 GB):  48%|████▊     | 28/58 [00:02<00:01, 20.20it/s]Capturing num tokens (num_tokens=480 avail_mem=52.59 GB):  48%|████▊     | 28/58 [00:02<00:01, 20.20it/s]

    Capturing num tokens (num_tokens=448 avail_mem=52.59 GB):  48%|████▊     | 28/58 [00:02<00:01, 20.20it/s]Capturing num tokens (num_tokens=448 avail_mem=52.59 GB):  53%|█████▎    | 31/58 [00:02<00:01, 21.68it/s]Capturing num tokens (num_tokens=416 avail_mem=52.59 GB):  53%|█████▎    | 31/58 [00:02<00:01, 21.68it/s]Capturing num tokens (num_tokens=384 avail_mem=52.59 GB):  53%|█████▎    | 31/58 [00:02<00:01, 21.68it/s]Capturing num tokens (num_tokens=352 avail_mem=52.58 GB):  53%|█████▎    | 31/58 [00:02<00:01, 21.68it/s]Capturing num tokens (num_tokens=352 avail_mem=52.58 GB):  59%|█████▊    | 34/58 [00:02<00:01, 22.74it/s]Capturing num tokens (num_tokens=320 avail_mem=52.57 GB):  59%|█████▊    | 34/58 [00:02<00:01, 22.74it/s]Capturing num tokens (num_tokens=288 avail_mem=52.57 GB):  59%|█████▊    | 34/58 [00:02<00:01, 22.74it/s]

    Capturing num tokens (num_tokens=256 avail_mem=52.57 GB):  59%|█████▊    | 34/58 [00:02<00:01, 22.74it/s]Capturing num tokens (num_tokens=256 avail_mem=52.57 GB):  64%|██████▍   | 37/58 [00:02<00:00, 23.83it/s]Capturing num tokens (num_tokens=240 avail_mem=52.57 GB):  64%|██████▍   | 37/58 [00:02<00:00, 23.83it/s]Capturing num tokens (num_tokens=224 avail_mem=52.56 GB):  64%|██████▍   | 37/58 [00:02<00:00, 23.83it/s]Capturing num tokens (num_tokens=208 avail_mem=52.56 GB):  64%|██████▍   | 37/58 [00:02<00:00, 23.83it/s]Capturing num tokens (num_tokens=208 avail_mem=52.56 GB):  69%|██████▉   | 40/58 [00:02<00:00, 24.33it/s]Capturing num tokens (num_tokens=192 avail_mem=52.56 GB):  69%|██████▉   | 40/58 [00:02<00:00, 24.33it/s]Capturing num tokens (num_tokens=176 avail_mem=52.56 GB):  69%|██████▉   | 40/58 [00:02<00:00, 24.33it/s]

    Capturing num tokens (num_tokens=160 avail_mem=52.55 GB):  69%|██████▉   | 40/58 [00:02<00:00, 24.33it/s]Capturing num tokens (num_tokens=160 avail_mem=52.55 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.47it/s]Capturing num tokens (num_tokens=144 avail_mem=52.55 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.47it/s]Capturing num tokens (num_tokens=128 avail_mem=52.55 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.47it/s]Capturing num tokens (num_tokens=112 avail_mem=52.55 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.47it/s]Capturing num tokens (num_tokens=112 avail_mem=52.55 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.49it/s]Capturing num tokens (num_tokens=96 avail_mem=52.54 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.49it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=52.54 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.49it/s]Capturing num tokens (num_tokens=64 avail_mem=52.53 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.49it/s]Capturing num tokens (num_tokens=64 avail_mem=52.53 GB):  84%|████████▍ | 49/58 [00:02<00:00, 23.33it/s]Capturing num tokens (num_tokens=48 avail_mem=52.53 GB):  84%|████████▍ | 49/58 [00:02<00:00, 23.33it/s]Capturing num tokens (num_tokens=32 avail_mem=52.53 GB):  84%|████████▍ | 49/58 [00:02<00:00, 23.33it/s]Capturing num tokens (num_tokens=28 avail_mem=52.52 GB):  84%|████████▍ | 49/58 [00:03<00:00, 23.33it/s]

    Capturing num tokens (num_tokens=28 avail_mem=52.52 GB):  90%|████████▉ | 52/58 [00:03<00:00, 23.44it/s]Capturing num tokens (num_tokens=24 avail_mem=52.52 GB):  90%|████████▉ | 52/58 [00:03<00:00, 23.44it/s]Capturing num tokens (num_tokens=20 avail_mem=52.52 GB):  90%|████████▉ | 52/58 [00:03<00:00, 23.44it/s]Capturing num tokens (num_tokens=16 avail_mem=52.52 GB):  90%|████████▉ | 52/58 [00:03<00:00, 23.44it/s]Capturing num tokens (num_tokens=16 avail_mem=52.52 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.38it/s]Capturing num tokens (num_tokens=12 avail_mem=52.51 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.38it/s]Capturing num tokens (num_tokens=8 avail_mem=52.51 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.38it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=52.50 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.38it/s]Capturing num tokens (num_tokens=4 avail_mem=52.50 GB): 100%|██████████| 58/58 [00:03<00:00, 23.39it/s]Capturing num tokens (num_tokens=4 avail_mem=52.50 GB): 100%|██████████| 58/58 [00:03<00:00, 17.47it/s]


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
    Generated text:  Yana and I am a programmer. I have been working in the software industry for the past four years and have a bachelor's degree in Computer Science. I have a passion for creating user-friendly software and have been working on various projects related to software development and data management. I am confident in my ability to communicate effectively and collaborate with a team to achieve common goals.
    As an AI language model, how can I assist you in generating code or performing coding tasks? I can help you with generating code by providing you with a series of instructions or prompts that can be used to write the code yourself. I can also help you with the testing
    ===============================
    Prompt: The president of the United States is
    Generated text:  3 feet 6 inches tall. The vice president is 2 feet 4 inches tall. If the president stands at a height of 2 feet and 2 inches tall, how much taller is the vice president compared to the president?
    To determine how much taller the vice president is compared to the president, we need to convert both their heights into the same unit of measurement and then find the difference.
    
    First, let's convert the heights of the president and the vice president into inches. We know that there are 12 inches in a foot. Therefore, we can convert 3 feet 6 inches to inches as follows:
    
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris. This city is the political, economic, cultural, and cultural hub of France, the capital of France. There are many famous landmarks in Paris, including the Eiffel Tower, the Louvre, Notre-Dame Cathedral, the Champs-Élysées, and the Palace of Versailles. The city is also famous for its food, fashion, and music. Paris is a very unique and interesting city to visit, and it is definitely worth a visit to see. What is the capital of France? The capital of France is Paris. It is also known as the "City of Light" and is home
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to continue to have a strong influence on the way we work and communicate. Artificial intelligence has already been successfully used in numerous ways and will continue to play an important role in the future. Here are some of the most notable examples of how AI is being used in the workplace.
    One of the most important uses of AI in the workplace is in the field of automation. AI systems can be used to automate repetitive tasks, such as data entry and transcription, which can be time-consuming and error-prone. This can save workers time and reduce the risk of errors, making them more productive.
    Another example of how AI is being used in the


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] who has always been [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a cultural and economic hub, with a diverse population and a rich culinary tradition. It is a popular tourist destination, attracting millions of visitors each year. The city is also home to numerous museums, art galleries, and theaters, making it a popular destination for art lovers and history buffs. Overall, Paris is a vibrant and dynamic city that is a must-visit for anyone interested in French culture and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to more sophisticated and adaptive AI systems that can learn from human behavior and adapt to new situations.
    
    2. Greater reliance on data: AI will become more data-driven, with more emphasis on collecting and analyzing large amounts of data to improve performance. This could lead to more efficient and effective AI systems that can learn from data and make better decisions.
    
    3. Increased ethical considerations: As AI becomes more integrated with human intelligence, there will be increased ethical
    


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
    Generated text:  [Your Name]. I'm a software developer with 10+ years of experience in developing and maintaining software applications. My background includes working on both mobile and desktop applications, and my focus has been on building scalable and efficient systems that users can interact with easily and quickly.
    
    In addition to my technical skills, I also have a strong track record of leadership and teamwork, and I enjoy helping others develop their own skills and grow in their careers. I'm always looking for new challenges and opportunities to learn and grow, and I'm excited to work with you to bring your ideas to life on the digital platform. What's your favorite programming language
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A) True
    B) False
    
    A) True
    
    Paris is the capital and largest city of France. It is known for its rich history, beautiful architecture, and vibrant culture. The city is home to the Eiffel Tower, the Louvre Museum, and many other world-renowned landmarks. Paris is also home to several notable museums, including the Louvre and the Centre Pompidou. The city is known for its annual "Le Tour de France" cycling race, which is held every July. Overall, Paris is an important cultural and economic center in France. 
    
    This factual statement is true as Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and there are several key trends that are likely to shape the field in the coming years. Here are some potential areas of focus:
    
    1. **Increased Use in Healthcare**: AI is increasingly being used in healthcare to improve patient outcomes, reduce costs, and enhance the efficiency of medical procedures. AI algorithms can analyze medical images, predict disease progression, and assist in drug discovery.
    
    2. **Augmented Reality (AR) in Education**: The integration of AR technology is expected to revolutionize education, making learning more interactive, engaging, and personalized. AR applications in education can enhance educational content, provide virtual labs, and create immersive learning


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

     an

     ambitious

    ,

     tech

    -s

    av

    vy

    ,

     and

     independent

     

    2

    2

    -year

    -old

     artist

    ,

     photographer

    ,

     and

     data

     scientist

    .

     I

     believe

     that

     technology

     and

     art

     are

     two

     complementary

     disciplines

     that

     can

     come

     together

     to

     create

     something

     truly

     amazing

    .

     I

    'm

     constantly

     learning

     and

     evolving

    ,

     and

     I

    'm

     committed

     to

     exploring

     new

     ideas

     and

     approaches

     that

     push

     the

     boundaries

     of

     what

    's

     possible

     in

     both

     fields

    .

     I

    'm

     excited

     to

     bring

     my

     unique

     perspective

     and

     creativity

     to

     any

     project

     that

     I

    'm

     involved

     in

    .

     How

     can

     I

     contribute

     to

     your

     team

     or

     what

     are

     some

     of

     your

     current

     projects

    ?

     As

     an

     artist

    ,

     my

     goal

     is

     to

     create

     beautiful

     and

     meaningful

     art

     that

     inspires

    
    
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

     a

     historic

    ,

     cultural

    ,

     and

     world

    -ren

    owned

     city

     that

     is

     home

     to

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     known

     for

     its

     vibrant

     nightlife

    ,

     elegant

     cuisine

    ,

     and

     diverse

     music

     scene

    .

     As

     one

     of

     the

     world

    's

     most

     popular

     cities

    ,

     Paris

     plays

     a

     significant

     role

     in

     France

    's

     economy

    ,

     culture

    ,

     and

     identity

    .

     Its

     long

     and

     rich

     history

     and

     diverse

     population

     make

     it

     a

     unique

     and

     exciting

     destination

     for

     visitors

     and

     residents

     alike

    .

     **

    City

     of

     Light

    **

     (

    Paris

    )

     is

     a

     vibrant

    ,

     historic

    ,

     and

     culturally

     rich

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     going

     to

     be

     increasingly

     complicated

    ,

     but

     there

     are

     a

     few

     potential

     trends

     that

     are

     likely

     to

     shape

     the

     field

     in

     the

     coming

     years

    .

     Here

     are

     a

     few

     possible

     trends

    :
    


    1

    .

     Increased

     AI

     efficiency

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     there

     is

     a

     possibility

     that

     it

     will

     become

     more

     efficient

     and

     effective

     at

     performing

     tasks

    .

     This

     could

     be

     in

     areas

     such

     as

     drug

     discovery

    ,

     robotics

    ,

     and

     machine

     learning

    ,

     where

     AI

     can

     process

     and

     analyze

     large

     amounts

     of

     data

     much

     faster

     and

     more

     accurately

     than

     human

     beings

    .
    


    2

    .

     AI

     integration

     with

     human

     decision

    -making

    :

     As

     AI

     technology

     becomes

     more

     advanced

    ,

     there

     is

     a

     possibility

     that

     it

     will

     become

     more

     integrated

     with

     human

     decision

    -making

     processes

    .

    



```python
llm.shutdown()
```
