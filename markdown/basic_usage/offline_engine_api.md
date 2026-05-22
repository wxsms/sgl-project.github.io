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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.15it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:44,  1.20it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:44,  1.20it/s]

    Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:44,  1.20it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:25,  2.04it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:25,  2.04it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:25,  2.04it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:25,  2.04it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:13,  3.66it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:13,  3.66it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:13,  3.66it/s]

    Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:13,  3.66it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:13,  3.66it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:07,  6.42it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:07,  6.42it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:07,  6.42it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:07,  6.42it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:07,  6.42it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:04<00:07,  6.42it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:03, 10.52it/s]

    Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:03, 10.52it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:04<00:03, 10.52it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 14.93it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 14.93it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 14.93it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 14.93it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 14.93it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 14.93it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 19.63it/s]

    Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 19.63it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 26.40it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 26.40it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 26.40it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 26.40it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 26.40it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:00, 26.40it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:00, 26.40it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 32.71it/s]

    Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 32.71it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 32.71it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 39.34it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 39.34it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 39.34it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 39.34it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 39.34it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 39.34it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 39.34it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 39.34it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 45.90it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 45.90it/s]

    Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 45.90it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 45.90it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 45.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.54 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.54 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.54 GB):   3%|▎         | 2/58 [00:00<00:03, 15.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.54 GB):   3%|▎         | 2/58 [00:00<00:03, 15.12it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.53 GB):   3%|▎         | 2/58 [00:00<00:03, 15.12it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.53 GB):   7%|▋         | 4/58 [00:00<00:03, 16.62it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.51 GB):   7%|▋         | 4/58 [00:00<00:03, 16.62it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.49 GB):   7%|▋         | 4/58 [00:00<00:03, 16.62it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.48 GB):   7%|▋         | 4/58 [00:00<00:03, 16.62it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.48 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.48 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.49 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.12it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=53.49 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.49 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.48 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.48 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.47 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.46 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.37it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.37it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=53.40 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.39 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.38 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.31it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=53.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.37 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.42it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.35 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.42it/s]

    Capturing num tokens (num_tokens=960 avail_mem=53.36 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.42it/s] Capturing num tokens (num_tokens=960 avail_mem=53.36 GB):  38%|███▊      | 22/58 [00:01<00:02, 14.92it/s]Capturing num tokens (num_tokens=896 avail_mem=53.35 GB):  38%|███▊      | 22/58 [00:01<00:02, 14.92it/s]Capturing num tokens (num_tokens=832 avail_mem=53.34 GB):  38%|███▊      | 22/58 [00:01<00:02, 14.92it/s]

    Capturing num tokens (num_tokens=832 avail_mem=53.34 GB):  41%|████▏     | 24/58 [00:01<00:02, 14.34it/s]Capturing num tokens (num_tokens=768 avail_mem=53.34 GB):  41%|████▏     | 24/58 [00:01<00:02, 14.34it/s]Capturing num tokens (num_tokens=704 avail_mem=53.33 GB):  41%|████▏     | 24/58 [00:01<00:02, 14.34it/s]Capturing num tokens (num_tokens=704 avail_mem=53.33 GB):  45%|████▍     | 26/58 [00:01<00:02, 13.76it/s]Capturing num tokens (num_tokens=640 avail_mem=53.33 GB):  45%|████▍     | 26/58 [00:01<00:02, 13.76it/s]

    Capturing num tokens (num_tokens=576 avail_mem=53.32 GB):  45%|████▍     | 26/58 [00:01<00:02, 13.76it/s]Capturing num tokens (num_tokens=576 avail_mem=53.32 GB):  48%|████▊     | 28/58 [00:01<00:02, 13.44it/s]Capturing num tokens (num_tokens=512 avail_mem=53.30 GB):  48%|████▊     | 28/58 [00:01<00:02, 13.44it/s]Capturing num tokens (num_tokens=480 avail_mem=53.32 GB):  48%|████▊     | 28/58 [00:01<00:02, 13.44it/s]

    Capturing num tokens (num_tokens=480 avail_mem=53.32 GB):  52%|█████▏    | 30/58 [00:01<00:02, 13.47it/s]Capturing num tokens (num_tokens=448 avail_mem=53.31 GB):  52%|█████▏    | 30/58 [00:01<00:02, 13.47it/s]Capturing num tokens (num_tokens=416 avail_mem=53.31 GB):  52%|█████▏    | 30/58 [00:01<00:02, 13.47it/s]Capturing num tokens (num_tokens=384 avail_mem=53.30 GB):  52%|█████▏    | 30/58 [00:01<00:02, 13.47it/s]Capturing num tokens (num_tokens=384 avail_mem=53.30 GB):  57%|█████▋    | 33/58 [00:02<00:01, 16.04it/s]Capturing num tokens (num_tokens=352 avail_mem=53.29 GB):  57%|█████▋    | 33/58 [00:02<00:01, 16.04it/s]Capturing num tokens (num_tokens=320 avail_mem=53.29 GB):  57%|█████▋    | 33/58 [00:02<00:01, 16.04it/s]Capturing num tokens (num_tokens=288 avail_mem=53.28 GB):  57%|█████▋    | 33/58 [00:02<00:01, 16.04it/s]Capturing num tokens (num_tokens=256 avail_mem=53.28 GB):  57%|█████▋    | 33/58 [00:02<00:01, 16.04it/s]

    Capturing num tokens (num_tokens=240 avail_mem=53.28 GB):  57%|█████▋    | 33/58 [00:02<00:01, 16.04it/s]Capturing num tokens (num_tokens=240 avail_mem=53.28 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.33it/s]Capturing num tokens (num_tokens=224 avail_mem=53.27 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.33it/s]Capturing num tokens (num_tokens=208 avail_mem=53.27 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.33it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  66%|██████▌   | 38/58 [00:02<00:00, 23.33it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  71%|███████   | 41/58 [00:02<00:00, 21.58it/s]Capturing num tokens (num_tokens=176 avail_mem=70.92 GB):  71%|███████   | 41/58 [00:02<00:00, 21.58it/s]

    Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:02<00:00, 21.58it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:02<00:00, 21.58it/s]Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:02<00:00, 21.58it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:02<00:00, 21.58it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  79%|███████▉  | 46/58 [00:02<00:00, 27.65it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:02<00:00, 27.65it/s] Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:02<00:00, 27.65it/s]Capturing num tokens (num_tokens=64 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:02<00:00, 27.65it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:02<00:00, 27.65it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:02<00:00, 27.65it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.27it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.27it/s]

    Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.27it/s]Capturing num tokens (num_tokens=20 avail_mem=70.88 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.27it/s]Capturing num tokens (num_tokens=16 avail_mem=70.88 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.27it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.27it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  97%|█████████▋| 56/58 [00:02<00:00, 35.80it/s]Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  97%|█████████▋| 56/58 [00:02<00:00, 35.80it/s] Capturing num tokens (num_tokens=4 avail_mem=70.87 GB):  97%|█████████▋| 56/58 [00:02<00:00, 35.80it/s]Capturing num tokens (num_tokens=4 avail_mem=70.87 GB): 100%|██████████| 58/58 [00:02<00:00, 21.90it/s]


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
    Generated text:  Veronica, I'm 22 years old and I just moved into a new house. I've been living alone, but I like spending time with my friends. My friend Sophie is also living in the same house, but she wants to move out and spend more time with her family. We both have our own hobbies and interests, but our schedules are quite different. I'm not very good at making plans or organizing my schedule, and I have trouble managing my time. Sophie, on the other hand, is very organized and has a keen sense of time management. 
    
    When it comes to my hobbies and interests, I enjoy spending time
    ===============================
    Prompt: The president of the United States is
    Generated text:  36 years older than 2 times the age of a woman. If the president is currently 58 years old, what is the woman's current age?
    To determine the woman's current age, we need to follow the given relationship step by step.
    
    1. Let's denote the president's current age as \( P \).
    2. According to the problem, the president is 36 years older than 2 times the age of the woman. This can be expressed mathematically as:
       \[
       P = 2 \times \text{woman's age} + 36
       \]
    3
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Rome
    C. Athens
    D. Madrid
    Answer:
    
    A
    
    The main purpose of the passage is to ______.
    A. present a summary of information
    B. present information in a chronological order
    C. provide evidence for an argument
    D. suggest a way to achieve an outcome
    Answer:
    
    C
    
    Which of the following statements about the provincial administrative regions of China is true?
    A. Tibet Autonomous Region and Xinjiang Uygur Autonomous Region are located in the northwest region.
    B. Beijing is the capital of China and is located in the southeast region.
    C. The provincial administrative regions
    ===============================
    Prompt: The future of AI is
    Generated text:  in the realm of algorithms, and the field of automation is a trend that is already moving in that direction. However, there are some areas where the use of AI is being seriously limited by the way we design it.
    For example, as the world becomes more and more interconnected, it is becoming increasingly difficult to predict what is going to happen next. This is where the use of AI can help to make the world safer by automating certain tasks and reducing the number of human error.
    However, it is also important to consider the ethical implications of using AI in a way that goes beyond the use of simple algorithms. One example of this is


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and opera houses. Paris is a popular tourist destination, known for its rich history, art, and cuisine. It is also home to the French Parliament and the French Academy of Sciences. The city is known for its diverse population, including French, African, and immigrant communities. Paris is a vibrant and dynamic city with a rich history and a strong sense of community. Its status as the capital of France is a testament to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more natural and intuitive interactions between humans and machines.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more transparent and accountable AI systems that are designed to minimize harm and maximize benefits.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes
    


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
    Generated text:  [Name] and I am a [field of work] with experience [number of years working in this field]. My specialties include [mention specific skills or knowledge relevant to your field of work]. I have a keen eye for detail and am able to communicate effectively with people of all backgrounds and levels. My work ethic is strong, and I am always looking for ways to improve my skills and knowledge. Thank you for taking the time to learn more about me. [Name] with interest. What is your field of work and what are your specialties? Dear [Name], I am [Name] and I am a [field of work]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a bustling metropolis with a rich history and cultural significance. With iconic landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum, Paris is a popular tourist destination and a cultural hub for the country. Its architecture, cuisine, and traditions have influenced much of European culture and are celebrated by visitors to the city. The city is also home to many important organizations such as the French National Library and the European Parliament. Overall, Paris is a city of diversity and importance that continues to draw people from all over the world. 
    
    This statement encapsulates the key facts about Paris as described in the passage
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid advancement, integration, and diverse applications across various industries. Here are some possible future trends in AI:
    
    1. Automation: AI is expected to revolutionize automation in manufacturing, transportation, and customer service. Automation can reduce human errors, increase efficiency, and improve productivity.
    
    2. Ethical AI: As AI becomes more integrated into our daily lives, there will be concerns about its ethics and potential biases. This is leading to the development of ethical AI policies and standards.
    
    3. Personalization: AI can be used to personalize user experiences, from recommendations on social media to personalized marketing campaigns. This will enable businesses to


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

     Jane

    .

     I

     am

     a

     computer

     programmer

     with

     over

     

    1

    0

     years

     of

     experience

    .

     I

     have

     a

     knack

     for

     solving

     problems

     and

     helping

     others

    .

     I

     enjoy

     sharing

     my

     knowledge

     with

     others

     through

     my

     coding

     and

     I

     strive

     to

     continually

     improve

     my

     skills

    .

     What

     other

     skills

     do

     you

     have

    ?

     I

     am

     also

     a

     good

     communicator

     and

     I

     enjoy

     helping

     people

     navigate

     their

     own

     problems

    .

     I

     have

     a

     friendly

     personality

     and

     I

     enjoy

     social

    izing

    .

     My

     ultimate

     goal

     is

     to

     help

     others

     reach

     their

     full

     potential

    .

     I

     believe

     that

     success

     is

     not

     just

     about

     achieving

     your

     goals

     but

     also

     about

     creating

     and

     maintaining

     meaningful

     connections

     with

     others

    .

     I

     am

     always

     eager

     to

     learn

     and

     improve

    ,

     and

     I

     believe

     that

     always

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     landmarks

    ,

     fashion

    ,

     and

     cuisine

    .

     It

     is

     a

     major

     cultural

     and

     economic

     center

     and

     home

     to

     many

     of

     France

    ’s

     historical

     and

     artistic

     monuments

    .

     Paris

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

     and

     is

     famous

     for

     its

     tape

    st

    ries

    ,

     theaters

    ,

     and

     its

     

    2

    0

    1

    7

     World

     Cup

     football

     tournament

    .

     It

     is

     also

     the

     largest

     city

     in

     France

     and

     home

     to

     numerous

     international

     organizations

     and

     cultural

     institutions

    .

     The

     city

     is

     known

     for

     its

     annual

     "

    The

    atre

     of

     the

     W

    ards

    ,"

     which

     is

     a

     major

     festival

     that

     showcases

     the

     works

     of

     French

     artists

    .

     Paris

     is

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    ,

     known

     for

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     a

     rapidly

     evolving

     field

     with

     numerous

     potential

     trends

     that

     could

     shape

     the

     development

     of

     AI

     technology

    .

     Some

     of

     the

     key

     trends

     that

     are

     currently

     in

     the

     development

     phase

     and

     could

     impact

     the

     AI

     industry

     include

    :
    


    1

    .

     Increased

     privacy

     and

     security

    :

     As

     more

     and

     more

     AI

     is

     being

     used

     in

     daily

     life

    ,

     there

     is

     a

     growing

     concern

     about

     how

     to

     protect

     the

     privacy

     and

     security

     of

     the

     data

     that

     is

     being

     processed

    .

     There

     is

     a

     need

     for

     AI

     systems

     to

     be

     designed

     in

     a

     way

     that

     respects

     user

     privacy

     and

     is

     designed

     to

     protect

     data

     from

     malicious

     actors

    .
    


    2

    .

     AI

    -driven

     automation

    :

     AI

     is

     increasingly

     being

     used

     to

     automate

     repetitive

     and

     mundane

     tasks

    ,

     which

     could

     have

    



```python
llm.shutdown()
```
