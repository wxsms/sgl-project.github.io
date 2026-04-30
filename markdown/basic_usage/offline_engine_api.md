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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]


    2026-04-30 06:44:06,603 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 06:44:06] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:37,  4.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:37,  4.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:37,  4.86s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:12,  1.31s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:12,  1.31s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:05<01:12,  1.31s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:05<01:12,  1.31s/it]Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:05<01:12,  1.31s/it]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:22,  2.27it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:22,  2.27it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:22,  2.27it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:22,  2.27it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:05<00:22,  2.27it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:05<00:22,  2.27it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:05<00:22,  2.27it/s]Compiling num tokens (num_tokens=2816):  12%|█▏        | 7/58 [00:05<00:22,  2.27it/s]Compiling num tokens (num_tokens=2560):  12%|█▏        | 7/58 [00:05<00:22,  2.27it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:06,  6.29it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:06,  6.29it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:06,  6.29it/s]

    Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:06,  6.29it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:06,  6.29it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:06,  6.29it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  9.16it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  9.16it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  9.16it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  9.16it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  9.16it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  9.16it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  9.16it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  9.16it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  9.16it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 15.35it/s]

    Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 15.35it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:01, 20.25it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 24.61it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 24.61it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 24.61it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 24.61it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 24.61it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 24.61it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 24.61it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 29.80it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 29.80it/s]

    Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 34.92it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 34.92it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 34.92it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 34.92it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 34.92it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 34.92it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 34.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.21 GB):   3%|▎         | 2/58 [00:00<00:05, 10.82it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.20 GB):   3%|▎         | 2/58 [00:00<00:05, 10.82it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.19 GB):   3%|▎         | 2/58 [00:00<00:05, 10.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.19 GB):   7%|▋         | 4/58 [00:00<00:04, 11.71it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.19 GB):   7%|▋         | 4/58 [00:00<00:04, 11.71it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.18 GB):   7%|▋         | 4/58 [00:00<00:04, 11.71it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=59.18 GB):  10%|█         | 6/58 [00:00<00:04, 12.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.17 GB):  10%|█         | 6/58 [00:00<00:04, 12.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.16 GB):  10%|█         | 6/58 [00:00<00:04, 12.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.16 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.16 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.12it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=59.13 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.13 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.09 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.09 GB):  17%|█▋        | 10/58 [00:00<00:03, 12.10it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=59.09 GB):  21%|██        | 12/58 [00:00<00:03, 13.31it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.08 GB):  21%|██        | 12/58 [00:00<00:03, 13.31it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.09 GB):  21%|██        | 12/58 [00:00<00:03, 13.31it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.09 GB):  24%|██▍       | 14/58 [00:01<00:03, 14.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.58 GB):  24%|██▍       | 14/58 [00:01<00:03, 14.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.44 GB):  24%|██▍       | 14/58 [00:01<00:03, 14.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.43 GB):  24%|██▍       | 14/58 [00:01<00:03, 14.13it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.43 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.73it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.42 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.42 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.41 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.41 GB):  34%|███▍      | 20/58 [00:01<00:02, 17.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.39 GB):  34%|███▍      | 20/58 [00:01<00:02, 17.51it/s]

    Capturing num tokens (num_tokens=960 avail_mem=58.41 GB):  34%|███▍      | 20/58 [00:01<00:02, 17.51it/s] Capturing num tokens (num_tokens=960 avail_mem=58.41 GB):  38%|███▊      | 22/58 [00:01<00:02, 16.97it/s]Capturing num tokens (num_tokens=896 avail_mem=58.38 GB):  38%|███▊      | 22/58 [00:01<00:02, 16.97it/s]Capturing num tokens (num_tokens=832 avail_mem=58.39 GB):  38%|███▊      | 22/58 [00:01<00:02, 16.97it/s]Capturing num tokens (num_tokens=768 avail_mem=58.38 GB):  38%|███▊      | 22/58 [00:01<00:02, 16.97it/s]Capturing num tokens (num_tokens=768 avail_mem=58.38 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.57it/s]Capturing num tokens (num_tokens=704 avail_mem=58.38 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.57it/s]Capturing num tokens (num_tokens=640 avail_mem=58.37 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.57it/s]

    Capturing num tokens (num_tokens=576 avail_mem=58.37 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.57it/s]Capturing num tokens (num_tokens=512 avail_mem=58.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.57it/s]Capturing num tokens (num_tokens=512 avail_mem=58.35 GB):  50%|█████     | 29/58 [00:01<00:01, 23.42it/s]Capturing num tokens (num_tokens=480 avail_mem=58.36 GB):  50%|█████     | 29/58 [00:01<00:01, 23.42it/s]Capturing num tokens (num_tokens=448 avail_mem=58.35 GB):  50%|█████     | 29/58 [00:01<00:01, 23.42it/s]Capturing num tokens (num_tokens=416 avail_mem=58.35 GB):  50%|█████     | 29/58 [00:01<00:01, 23.42it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  50%|█████     | 29/58 [00:01<00:01, 23.42it/s]Capturing num tokens (num_tokens=384 avail_mem=58.34 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.41it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.41it/s]

    Capturing num tokens (num_tokens=320 avail_mem=58.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.41it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.41it/s]Capturing num tokens (num_tokens=256 avail_mem=58.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.41it/s]Capturing num tokens (num_tokens=256 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=240 avail_mem=58.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=224 avail_mem=58.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=208 avail_mem=58.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=192 avail_mem=58.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.45it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  64%|██████▍   | 37/58 [00:02<00:00, 29.45it/s]Capturing num tokens (num_tokens=176 avail_mem=58.30 GB):  72%|███████▏  | 42/58 [00:02<00:00, 33.58it/s]Capturing num tokens (num_tokens=160 avail_mem=58.29 GB):  72%|███████▏  | 42/58 [00:02<00:00, 33.58it/s]

    Capturing num tokens (num_tokens=144 avail_mem=58.29 GB):  72%|███████▏  | 42/58 [00:02<00:00, 33.58it/s]Capturing num tokens (num_tokens=128 avail_mem=58.29 GB):  72%|███████▏  | 42/58 [00:02<00:00, 33.58it/s]Capturing num tokens (num_tokens=112 avail_mem=58.29 GB):  72%|███████▏  | 42/58 [00:02<00:00, 33.58it/s]Capturing num tokens (num_tokens=96 avail_mem=58.28 GB):  72%|███████▏  | 42/58 [00:02<00:00, 33.58it/s] Capturing num tokens (num_tokens=96 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:02<00:00, 36.15it/s]Capturing num tokens (num_tokens=80 avail_mem=58.28 GB):  81%|████████  | 47/58 [00:02<00:00, 36.15it/s]Capturing num tokens (num_tokens=64 avail_mem=58.27 GB):  81%|████████  | 47/58 [00:02<00:00, 36.15it/s]Capturing num tokens (num_tokens=48 avail_mem=58.27 GB):  81%|████████  | 47/58 [00:02<00:00, 36.15it/s]Capturing num tokens (num_tokens=32 avail_mem=58.27 GB):  81%|████████  | 47/58 [00:02<00:00, 36.15it/s]Capturing num tokens (num_tokens=28 avail_mem=58.26 GB):  81%|████████  | 47/58 [00:02<00:00, 36.15it/s]

    Capturing num tokens (num_tokens=28 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:02<00:00, 38.26it/s]Capturing num tokens (num_tokens=24 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:02<00:00, 38.26it/s]Capturing num tokens (num_tokens=20 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:02<00:00, 38.26it/s]Capturing num tokens (num_tokens=16 avail_mem=58.26 GB):  90%|████████▉ | 52/58 [00:02<00:00, 38.26it/s]Capturing num tokens (num_tokens=12 avail_mem=58.25 GB):  90%|████████▉ | 52/58 [00:02<00:00, 38.26it/s]Capturing num tokens (num_tokens=8 avail_mem=58.25 GB):  90%|████████▉ | 52/58 [00:02<00:00, 38.26it/s] Capturing num tokens (num_tokens=8 avail_mem=58.25 GB):  98%|█████████▊| 57/58 [00:02<00:00, 39.94it/s]Capturing num tokens (num_tokens=4 avail_mem=58.24 GB):  98%|█████████▊| 57/58 [00:02<00:00, 39.94it/s]Capturing num tokens (num_tokens=4 avail_mem=58.24 GB): 100%|██████████| 58/58 [00:02<00:00, 24.01it/s]


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
    Generated text:  Kat, and I am a high school student majoring in Psychology. I am in the 8th grade and I am very interested in sports. I have been playing soccer and badminton for a few years, and I really enjoy them both. I also really want to be a professional player, and I want to play for a big club. I have been in a few leagues, and I have tried different styles of play. However, I have not found a club that fits my style of play. 
    
    My question is: what is the best way to find a good professional soccer player or badminton player, and what
    ===============================
    Prompt: The president of the United States is
    Generated text:  a president who has to be at least 35 years old. The president of the United States can be at most 70 years old. What is the range of ages of a president who is 35 years old?
    To determine the range of ages of a president who is 35 years old, we need to identify the lower and upper bounds of the age range for a president of the United States.
    
    1. Identify the minimum age limit:
       - The president must be at least 35 years old.
       - Therefore, the minimum age limit is 35 years old.
    
    2. Identify the maximum age
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The city has a history of over 600 years. The most famous building of the city is the Eiffel Tower. The city is quite a vibrant place. It's a place to do lots of things. Most of the French language can be heard in the city. The weather is usually mild and sunny. The temperatures in the city are usually around 10 to 20 degrees Celsius. The month with the most rainy days is January. The city has lots of parks and beaches. In the winter, you can visit the Louvre, the Eiffel Tower, and the Notre-Dame Cathedral.
    ===============================
    Prompt: The future of AI is
    Generated text:  not about the convergence of AI and human intelligence, but rather about the creation of a unified AI.
    In other words, the future of AI is about making the most of existing AI, not trying to create something entirely new. This approach can be achieved by creating a more cohesive and unified set of AI technologies that are able to learn and adapt to new and complex problems.
    The creation of a unified AI would involve a combination of existing AI technologies, such as machine learning, deep learning, and natural language processing, with the addition of new technologies that are specifically designed to address the unique challenges of the future.
    Some of the key technologies that could


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I have always been passionate about [Your Passion], and I am always looking for ways to [Your Goal]. I am always eager to learn and grow, and I am always willing to help others. I am a [Your Character Trait] and I am always ready to [Your Character Trait]. I am a [Your Character] and I am always [Your Character Trait]. I am a [Your Character] and I am always [Your Character Trait]. I am a [Your Character] and I am always [Your Character Trait]. I am a [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the country and the second-largest city in the European Union. It is located on the Seine River and is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its rich history, art, and culture, and is a popular tourist destination. The city is also home to many famous landmarks and attractions, including the Louvre Museum, the Champs-Élysées, and the Eiffel Tower. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage. Its status
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical AI: As more and more AI systems are being developed, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and fairness. AI systems that are designed to be transparent and accountable will be more likely to be accepted and used in a wider range of applications.
    
    2. Increased use of AI in healthcare: AI is already being used in healthcare to improve
    


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
    Generated text:  [Name] and I'm a [Age] year old AI assistant that's been developed by [Company]. I'm a virtual assistant designed to assist and serve as a resource for anyone who needs help with anything. I can answer basic questions, provide information on various topics, and even help with various tasks and projects. I'm constantly learning and improving my abilities, and I'm always here to assist you. How can I assist you today? [Name] is a neutral self-introduction written in the first person, with no gender or age information provided. The character's name, age, and company are fictional and not mentioned. The
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Calculation:
    1. Identify the main subject: "France"
    2. Identify the main location: "Capital"
    3. Identify the capital city: "Paris"
    4. Calculate the total word count: 3
    
    Final answer: Paris. 
    
    Calculation: 
    1. Subject: France
    2. Location: Capital
    3. Capital city: Paris
    4. Total word count: 3
    
    Final answer: Paris. 
    
    Calculation: 3 words. 
    
    The statement is concise, relevant, and accurate. Paris is the capital of France, and it is the largest city in the country. Its location
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  quite dynamic and will likely involve a number of trends that will shape how we use and interact with AI systems. Some of the key trends include:
    
    1. Increased focus on ethical AI: The increasing concern about AI's potential to cause harm and unintended consequences has led to increased focus on ethical AI. This includes developing AI that is designed with the goal of minimizing harm and maximizing benefits, as well as developing AI that is designed to be transparent, responsible, and accountable.
    
    2. Better understanding of AI: As AI systems become more complex and sophisticated, there is a greater need for better understanding of how they work and how they can be improved.


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

     name

    ].

     I

     am

     [

    insert

     profession

    ,

     age

    ,

     etc

    .

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     Here

     is

     a

     possible

     introduction

    :


    "

    Hello

    ,

     my

     name

     is

     [

    insert

     name

    ].

     I

     am

     [

    insert

     profession

    ,

     age

    ,

     etc

    .

    ].

     I

     am

     a

     [

    insert

     occupation

    ]

     with

     a

     love

     for

     [

    insert

     hobby

     or

     activity

    ].

     In

     my

     free

     time

    ,

     I

     enjoy

     [

    insert

     activity

     that

     makes

     me

     happy

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?"


    This

     introduction

     is

     neutral

     and

     doesn

    't

     explicitly

     mention

     a

     profession

     or

     occupation

    ,

     but

     it

     provides

     an

     overview

     of

     the

     character

    's

     identity

     and

     personal

     interests

    .

     It

    's

     a

     simple

     and

     effective

     way

     to

     start

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     and

     it

     is

     known

     for

     its

     stunning

     architecture

    ,

     vibrant

     art

     scene

    ,

     and

     delicious

     cuisine

    .

     It

     is

     also

     the

     country

    's

     cultural

     center

     and

     the

     location

     of

     numerous

     historic

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    .

     Additionally

    ,

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     home

     to

     many

     famous

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     Despite

     its

     reputation

     for

     luxury

     and

     sophistication

    ,

     Paris

     is

     also

     known

     for

     its

     frequent

     protests

     and

     political

     unrest

    ,

     which

     has

     made

     it

     a

     symbol

     of

     France

    's

     political

     and

     social

     conflicts

    .

     Overall

    ,

     Paris

     is

     a

     city

     of

     history

    ,

     culture

    ,

     and

     energy

     that

     is

     a

     must

    -

    visit

     for

     anyone

     visiting

     France

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     quite

     promising

    ,

     with

     many

     potential

     trends

     emerging

     as

     technologies

     advance

     and

     applications

     expand

    .

     Here

     are

     some

     of

     the

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     More

     precise

     and

     accurate

     AI

    :

     As

     AI

     continues

     to

     improve

    ,

     its

     ability

     to

     process

     and

     analyze

     data

     will

     become

     even

     more

     precise

     and

     accurate

    .

     This

     will

     allow

     for

     more

     accurate

     predictions

     and

     better

     decision

    -making

     in

     various

     fields

    ,

     including

     healthcare

    ,

     finance

    ,

     and

     transportation

    .
    


    2

    .

     More

     flexible

     and

     scalable

     AI

    :

     AI

     will

     become

     more

     flexible

     and

     scalable

    ,

     allowing

     for

     the

     implementation

     of

     new

     technologies

     and

     applications

    .

     This

     will

     enable

     the

     creation

     of

     AI

     systems

     that

     can

     adapt

     to

     changing

     conditions

     and

     handle

     increasingly

     complex

     problems

    .
    


    3

    .

     AI

     with

    



```python
llm.shutdown()
```
