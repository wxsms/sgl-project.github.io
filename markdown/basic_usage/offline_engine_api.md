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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.00it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:43,  4.98s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:43,  4.98s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:52,  1.03it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:52,  1.03it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:05<00:52,  1.03it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:05<00:52,  1.03it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:05<00:52,  1.03it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:19,  2.50it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:19,  2.50it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:19,  2.50it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:19,  2.50it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:19,  2.50it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:05<00:19,  2.50it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:05<00:19,  2.50it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:08,  5.45it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:08,  5.45it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:08,  5.45it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:08,  5.45it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:08,  5.45it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:05<00:08,  5.45it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:05<00:08,  5.45it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:05<00:08,  5.45it/s]

    Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:05<00:08,  5.45it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03, 10.45it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 16.39it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 16.39it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 16.39it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 16.39it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 16.39it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 16.39it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 16.39it/s]

    Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 16.39it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 16.39it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 23.31it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 23.31it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 32.22it/s]

    Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 32.22it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 43.03it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 43.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.36 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.36 GB):   3%|▎         | 2/58 [00:00<00:04, 12.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.36 GB):   3%|▎         | 2/58 [00:00<00:04, 12.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.35 GB):   3%|▎         | 2/58 [00:00<00:04, 12.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.35 GB):   7%|▋         | 4/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.35 GB):   7%|▋         | 4/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.34 GB):   7%|▋         | 4/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.34 GB):  10%|█         | 6/58 [00:00<00:03, 16.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.22 GB):  10%|█         | 6/58 [00:00<00:03, 16.42it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.32 GB):  10%|█         | 6/58 [00:00<00:03, 16.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.31 GB):  10%|█         | 6/58 [00:00<00:03, 16.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.31 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.53it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.31 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.30 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.53it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.29 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.53it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.29 GB):  21%|██        | 12/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.28 GB):  21%|██        | 12/58 [00:00<00:02, 21.30it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.26 GB):  21%|██        | 12/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.27 GB):  21%|██        | 12/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.27 GB):  21%|██        | 12/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.27 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.25 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.25it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.23 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.25it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.23 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.20 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.18it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.21 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.18it/s] Capturing num tokens (num_tokens=896 avail_mem=74.22 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.18it/s]Capturing num tokens (num_tokens=832 avail_mem=74.21 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.18it/s]Capturing num tokens (num_tokens=832 avail_mem=74.21 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.90it/s]Capturing num tokens (num_tokens=768 avail_mem=74.20 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.90it/s]Capturing num tokens (num_tokens=704 avail_mem=74.20 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.90it/s]Capturing num tokens (num_tokens=640 avail_mem=74.19 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.90it/s]Capturing num tokens (num_tokens=576 avail_mem=74.19 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.90it/s]Capturing num tokens (num_tokens=576 avail_mem=74.19 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.15it/s]Capturing num tokens (num_tokens=512 avail_mem=74.17 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.15it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.20 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.15it/s]Capturing num tokens (num_tokens=448 avail_mem=74.18 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.15it/s]Capturing num tokens (num_tokens=416 avail_mem=74.17 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.15it/s]Capturing num tokens (num_tokens=416 avail_mem=74.17 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=384 avail_mem=74.17 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=352 avail_mem=74.16 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=320 avail_mem=74.15 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=288 avail_mem=74.15 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=288 avail_mem=74.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=256 avail_mem=74.16 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.41it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=224 avail_mem=74.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=208 avail_mem=74.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=208 avail_mem=74.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.15it/s]Capturing num tokens (num_tokens=192 avail_mem=74.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.15it/s]Capturing num tokens (num_tokens=176 avail_mem=74.11 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.15it/s]Capturing num tokens (num_tokens=160 avail_mem=74.11 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.15it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.10 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.15it/s]Capturing num tokens (num_tokens=144 avail_mem=74.10 GB):  76%|███████▌  | 44/58 [00:01<00:00, 31.30it/s]Capturing num tokens (num_tokens=128 avail_mem=74.11 GB):  76%|███████▌  | 44/58 [00:01<00:00, 31.30it/s]Capturing num tokens (num_tokens=112 avail_mem=74.11 GB):  76%|███████▌  | 44/58 [00:01<00:00, 31.30it/s]Capturing num tokens (num_tokens=96 avail_mem=74.10 GB):  76%|███████▌  | 44/58 [00:01<00:00, 31.30it/s] Capturing num tokens (num_tokens=80 avail_mem=74.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 31.30it/s]Capturing num tokens (num_tokens=80 avail_mem=74.09 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.40it/s]Capturing num tokens (num_tokens=64 avail_mem=74.09 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.40it/s]Capturing num tokens (num_tokens=48 avail_mem=74.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.40it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.40it/s]Capturing num tokens (num_tokens=28 avail_mem=74.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.40it/s]Capturing num tokens (num_tokens=28 avail_mem=74.07 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.45it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.45it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.45it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 29.34it/s]


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
    Generated text:  John Smith. I have been a volunteer for over a decade. I have been a nurse at a hospital for many years. I have had a passion for healthcare for many years. I have been a nurse for the past 8 years. In the past, my job was to care for and monitor my patients, and to provide medical care and treatment. Now, I have a passion for teaching. My goal is to help and educate others on how to care for and keep patients healthy. I have been a nurse for many years. I have been a volunteer for over a decade, and I have a passion for healthcare. My goal is
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to use the word "banana" in a speech. The president decides to use "banana" once a month for a year. How many times will the president use the word "banana" in the year? To determine how many times the president will use the word "banana" in a year, we need to follow these steps:
    
    1. Identify the number of months in a year. A year has 12 months.
    2. Determine how many times the president uses the word "banana" per month. The president uses the word "banana" once a month.
    3. Calculate the total number of times the
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris. The city has always been a cultural center and was the seat of the kings of France for centuries. The people of Paris speak French, the official language of France. The capital of France is in the middle of the country. It is the capital of France but not in the country. France is a country with a lot of cities. The capital of France is very old. It is the oldest capital in Europe. For many years, the capital of France was not in Paris. When the kings of France needed to travel to the other cities in Europe, they had to travel to a city called Paris. But now
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but we must guard it with our eyes open. The future of AI is bright, but we must guard it with our eyes open.
    
    This is the opening of a report by the Council of Europe, which will outline a set of measures to help governments respond to the coming AI revolution.
    
    In a short report, the EU will outline a set of measures to help governments respond to the coming AI revolution. The EU is creating a task force to give the EU and other governments a broad view of the emerging technology. The task force will make recommendations on how AI could be handled by the EU. The EU is an EU that


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


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France, located on the Île de France, a small island in the Mediterranean Sea. It is the largest city in France and the second-largest city in Europe, with a population of over 2.1 million people. Paris is known for its rich history, art, and culture, and is a major tourist destination. The city is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a major hub for business, finance, and government, and is a popular tourist
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased use of AI in healthcare: AI is already being used in medical diagnosis and treatment, and we can expect to see even more sophisticated applications in the future.
    
    2. Integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. We can expect to see even more integration in the future, with AI becoming more integrated into our daily routines.
    
    3. AI in the workplace: AI is already being used in the
    


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
    Generated text:  [Name], and I'm an AI language model. I have been here for 10 years, and I've been around for as long as anyone has been alive. I have a few years of experience in the field of linguistics, and I'm always learning and improving. I'm here to provide you with information and help you with any questions you may have. Please let me know if you need anything. How can I assist you today? [Name] - Your AI language model. [Name] - Your friendly AI assistant. [Name] - Your impartial and unbiased AI assistant. [Name] - Your 10
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Ville Noire" (The Black City). It is the largest city in France and the third-largest city in the European Union. The city is home to many historical landmarks, including the Louvre Museum, Notre-Dame Cathedral, and the Eiffel Tower. It is known for its rich culture, arts, and cuisine, and is a popular tourist destination. As the birthplace of the French Revolution, Paris is also home to many important historical sites and monuments. The city is currently experiencing a significant cultural boom, with many French artists, musicians, and writers working in the region. 
    
    French cuisine
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be rapidly evolving with many possible trends and advancements. Some of the potential trends in AI include:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in our daily lives, there will be a growing emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and use.
    
    2. Emphasis on developing more advanced AI systems: As AI technology continues to advance, it is likely that we will see the development of more advanced AI systems with improved accuracy and capabilities.
    
    3. Increased use of AI in healthcare: AI could be used to improve the diagnosis and treatment of diseases, predict disease outbreaks,


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

     am

     a

     [

    occupation

    ]

     and

     [

    job

     title

    ].

     I

     am

     a

     [

    brief

     description

     of

     your

     job

    ]

     and

     am

     always

     available

     for

     [

    short

     description

     of

     your

     work

    ].

     I

    'm

     confident

     in

     my

     [

    strength

     or

     ability

    ]

     and

     have

     [

    number

     of

     years

     of

     experience

    ]

     years

     of

     experience

     in

     this

     field

    .

     I

     am

     dedicated

     to

     [

    reason

     why

     you

     are

     passionate

     about

     your

     work

    ].

     I

    'm

     passionate

     about

     [

    reason

     why

     you

     love

     your

     work

    ],

     and

     I

    'm

     always

     looking

     to

     learn

     new

     things

    .

     I

    'm

     excited

     to

     help

     you

     reach

     your

     goals

     and

     make

     a

     positive

     impact

     in

     your

     life

    .

     [

    Contact

     information

     for

     [

    you

     are

    ],

     if

     available

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     large

     and

     cosm

    opolitan

     city

     with

     a

     rich

     history

     and

     culture

    ,

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

    .

     Paris

     has

     a

     diverse

     population

     with

     French

    ,

     French

     Cre

    ole

    ,

     and

     international

     influences

    .

     The

     city

     is

     known

     for

     its

     famous

     landmarks

     such

     as

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

     Dame

     Cathedral

    ,

     and

     its

     vibrant

     nightlife

     and

     shopping

     scene

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     center

     of

     business

     and

     politics

     in

     France

    .

     The

     city

     has

     a

     history

     of

     involvement

     in

     the

     French

     Revolution

    ,

     the

     French

     Revolution

    ,

     and

     the

     French

     Revolution

    ,

     making

     it

     a

     hub

     of

     historical

     events

    .

     It

     is

     also

     known

     for

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     driven

     by

     several

     trends

     that

     could

     expand

     its

     capabilities

     and

     applications

    .

     Here

     are

     some

     potential

     future

     trends

     that

     could

     shape

     the

     AI

     landscape

    :
    


    1

    .

     Increased

     AI

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

    ,

     blockchain

    ,

     and

     blockchain

    -based

     applications

    ,

     the

     possibilities

     for

     AI

     integration

     are

     likely

     to

     expand

    .

     This

     integration

     could

     lead

     to

     more

     advanced

     and

     efficient

     AI

     systems

     that

     can

     work

     in

     conjunction

     with

     other

     technologies

    ,

     making

     it

     easier

     for

     humans

     to

     understand

     and

     utilize

     AI

    -powered

     solutions

    .
    


    2

    .

     AI

    -based

     medical

     advancements

    :

     AI

     is

     already

     being

     used

     in

     various

     medical

     applications

    ,

     such

     as

     patient

     diagnosis

    ,

     treatment

     planning

    ,

     and

     drug

     discovery

    



```python
llm.shutdown()
```
