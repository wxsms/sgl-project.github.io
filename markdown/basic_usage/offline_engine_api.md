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
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-26 08:07:12] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.74it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.73it/s]


    2026-04-26 08:07:16,456 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 08:07:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:43,  4.98s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:43,  4.98s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=5632):   2%|▏         | 1/58 [00:05<04:43,  4.98s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:32,  1.58it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:32,  1.58it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:05<00:32,  1.58it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:05<00:32,  1.58it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:05<00:32,  1.58it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:05<00:32,  1.58it/s]Compiling num tokens (num_tokens=3328):  10%|█         | 6/58 [00:05<00:32,  1.58it/s]

    Compiling num tokens (num_tokens=3072):  10%|█         | 6/58 [00:05<00:32,  1.58it/s]Compiling num tokens (num_tokens=2816):  10%|█         | 6/58 [00:05<00:32,  1.58it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:09,  4.56it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:09,  4.56it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:09,  4.56it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:09,  4.56it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:09,  4.56it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:05<00:09,  4.56it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:05<00:09,  4.56it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:05<00:09,  4.56it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:05<00:09,  4.56it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:04,  8.42it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.29it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.29it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.29it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.29it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.29it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.29it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.29it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.29it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.29it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:01, 19.20it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:01, 19.20it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:01, 19.20it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:01, 19.20it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:01, 19.20it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:01, 19.20it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:01, 19.20it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:01, 19.20it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:01, 19.20it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:01, 19.20it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 26.93it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 26.93it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 36.57it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 36.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.39 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.36 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.36 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.76it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.76it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.76it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=116.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.76it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.34 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.34 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.33 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.33 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.33 GB):  31%|███       | 18/58 [00:00<00:01, 35.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.33 GB):  31%|███       | 18/58 [00:00<00:01, 35.94it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.33 GB):  31%|███       | 18/58 [00:00<00:01, 35.94it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.31 GB):  31%|███       | 18/58 [00:00<00:01, 35.94it/s]

    Capturing num tokens (num_tokens=960 avail_mem=116.32 GB):  31%|███       | 18/58 [00:00<00:01, 35.94it/s] Capturing num tokens (num_tokens=896 avail_mem=116.32 GB):  31%|███       | 18/58 [00:00<00:01, 35.94it/s]Capturing num tokens (num_tokens=896 avail_mem=116.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=832 avail_mem=116.31 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=768 avail_mem=116.31 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=704 avail_mem=116.31 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=640 avail_mem=116.30 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=576 avail_mem=116.30 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=576 avail_mem=116.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=512 avail_mem=116.29 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=480 avail_mem=116.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=448 avail_mem=116.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.67it/s]

    Capturing num tokens (num_tokens=416 avail_mem=116.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=384 avail_mem=116.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.67it/s]Capturing num tokens (num_tokens=384 avail_mem=116.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.18it/s]Capturing num tokens (num_tokens=352 avail_mem=116.29 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.18it/s]Capturing num tokens (num_tokens=320 avail_mem=116.29 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.18it/s]Capturing num tokens (num_tokens=288 avail_mem=116.28 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.18it/s]Capturing num tokens (num_tokens=256 avail_mem=116.28 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.18it/s]Capturing num tokens (num_tokens=240 avail_mem=116.28 GB):  57%|█████▋    | 33/58 [00:01<00:00, 42.18it/s]Capturing num tokens (num_tokens=240 avail_mem=116.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=224 avail_mem=116.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=208 avail_mem=116.27 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.27it/s]

    Capturing num tokens (num_tokens=192 avail_mem=116.27 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=176 avail_mem=116.27 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=160 avail_mem=116.27 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=160 avail_mem=116.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=144 avail_mem=116.26 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=128 avail_mem=116.26 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=112 avail_mem=116.26 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=96 avail_mem=116.25 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.06it/s] Capturing num tokens (num_tokens=80 avail_mem=116.25 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=80 avail_mem=116.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=64 avail_mem=116.24 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.92it/s]

    Capturing num tokens (num_tokens=48 avail_mem=116.24 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=32 avail_mem=116.24 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=28 avail_mem=116.23 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=24 avail_mem=116.23 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=24 avail_mem=116.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=20 avail_mem=116.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=16 avail_mem=116.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=12 avail_mem=116.22 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=8 avail_mem=116.22 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.10it/s] Capturing num tokens (num_tokens=4 avail_mem=116.22 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.10it/s]

    Capturing num tokens (num_tokens=4 avail_mem=116.22 GB): 100%|██████████| 58/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=4 avail_mem=116.22 GB): 100%|██████████| 58/58 [00:01<00:00, 39.17it/s]


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
    Generated text:  x and I am an English teacher. My name is Student name. We are both students.
    I am very good at English and I want to learn more about the world. I have a dream to travel to the moon. I have already planned a 6-month trip.
    What would you do to help me go to the moon? (prompt)
    You can send me an email, you can use the message board, or you can ask me on Twitter. (prompt)
    
    It would be good to ask for your help to learn about the moon and to have the best experience on the trip. I am interested in learning about the solar system
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking executive of the federal government of the United States. The president, along with the vice president, the first lady, the cabinet of the president, the Secretary of State, the Secretary of Defense, and the Secretaries of other federal agencies, is the highest-ranking executive of the federal government. The president serves a 4-year term, and can be re-elected. A presidential election is held every four years. The president is also the commander-in-chief of the United States Armed Forces.
    
    Based on that paragraph can we conclude that the sentence "The president has no power to make any decisions" is true?
    Choices:
    A).
    ===============================
    Prompt: The capital of France is
    Generated text:  (　　)  
    A: Paris  
    B: London  
    C: Moscow  
    D: Berlin
    To determine the capital of France, let's analyze the options provided:
    
    A: Paris - This is a well-known French city and is the capital of France.
    B: London - This is the capital of England, not France.
    C: Moscow - This is the capital of Russia, not France.
    D: Berlin - This is the capital of Germany, not France.
    
    Since the capital of France is Paris, the correct answer is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  revolutionary, but it needs the right people to lead the way. Most of our current AI is set up to work in an impersonal, automated way, the kind that most people do not like. But, as a company, we want our AI to be autonomous, proactive, and able to help people find solutions to the challenges that they face.
    
    We want to help people have a better understanding of their own conditions and circumstances, and to see how their actions can help them better manage their lives. By doing this, we hope to help people improve their quality of life.
    
    To achieve this, we will need to hire a team of people


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am passionate about [job title] and have been working in the field for [number of years] years. I am always looking for new challenges and opportunities to grow and learn. I am a [job title] who is always looking for ways to improve my skills and knowledge. I am a [job title] who is always looking for ways to improve my skills and knowledge. I am a [job title] who is always looking for ways to improve my skills and knowledge. I am a [job title] who is always looking for ways to improve
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many of the world's most famous museums and attractions. Paris is a vibrant and diverse city with a rich history and a strong sense of French identity. It is a popular tourist destination and a major economic and cultural center in Europe. The city is also known for its cuisine, fashion, and music, and is home to many of the world's most famous artists and musicians. Paris is a city of contrasts,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are already being used in a wide range of applications, from self-driving cars to personalized medicine to fraud detection. As these technologies continue to evolve, we can expect to see even more innovative applications and improvements in AI. Additionally, there is a growing focus on ethical considerations and the responsible use of AI, as concerns about bias, privacy, and transparency continue to grow. Overall, the future of AI is likely to be characterized by continued innovation, growth, and development, with a focus on ethical and responsible use.
    


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
    Generated text:  [Name] and I'm a [Age] year old [Occupation] [Reason for being here] [Brief summary of your character's background and experiences].
    My goal is to be a [desired profession] and to share my experiences with others. I believe that learning about others, their experiences, and values is crucial in creating a better understanding of the world we live in. I am eager to share my knowledge and passion for learning with others, and I believe that together, we can make a positive impact on the world. What other information do you need to know about me? (Please don't give any personal details) 
    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital and largest city of France, located on the river Seine in the Centre-Val de Loire region. It is one of the most populated cities in the world, with over 2. 7 million inhabitants as of 2021. The city is home to the country's cultural and political center, with many important historical and artistic landmarks. It is also known as the "City of Light" and has been an international center for many artistic and intellectual movements throughout history. Paris is a UNESCO World Heritage site, and it is home to some of the world's most renowned museums, including the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dependent on several factors, including the level of development in hardware, software, and data. Here are some possible trends that may emerge in the coming years:
    
    1. Increased use of AI for healthcare: AI is already being used in healthcare to diagnose diseases, track patients, and optimize treatment plans. As more data is collected and AI algorithms become more sophisticated, it's possible that we'll see even more widespread use of AI in healthcare in the future.
    
    2. Development of AI-powered cognitive assist devices: As AI becomes more sophisticated, we may see the development of AI-powered cognitive assist devices that can help people with everyday tasks, such as


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

    'm

     a

     [

    job

     title

    ]

     at

     [

    Company

     Name

    ].

     I

    'm

     excited

     to

     start

     here

     and

     learn

     something

     new

    .

     What

     can

     I

     expect

     from

     working

     with

     you

    ?

     Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

    'm

     a

     [

    job

     title

    ]

     at

     [

    Company

     Name

    ].

     I

    'm

     excited

     to

     start

     here

     and

     learn

     something

     new

    .

     What

     can

     I

     expect

     from

     working

     with

     you

    ?

     Well

    ,

     hello

     there

    ,

     my

     name

     is

     [

    Name

    ]

     and

     I

    'm

     a

     [

    job

     title

    ]

     at

     [

    Company

     Name

    ].

     I

    'm

     excited

     to

     start

     here

     and

     learn

     something

     new

    .

     What

     can

     I

     expect

     from

     working

     with

     you

    ?

     Hello

    ,

     my

     name

     is

     [

    Name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Notre

     Dame

     Cathedral

    .

     It

     is

     also

     home

     to

     numerous

     world

    -class

     museums

    ,

     music

     festivals

    ,

     and

     art

     galleries

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

     French

     culture

     and

     cuisine

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     diverse

     culture

    ,

     and

     world

    -class

     architecture

    .

     According

     to

     a

     

    2

    0

    2

    1

     Forbes

     report

    ,

     Paris

     is

     the

     eighth

    -most

     crowded

     city

     in

     the

     world

    .

     The

     city

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

     Notre

     Dame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Other

     notable

     landmarks

     in

     Paris

     include

     the

     Arc

     de

     Tri

    omp

    he

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     shaped

     by

     a

     number

     of

     factors

    ,

     including

     advances

     in

     machine

     learning

    ,

     deep

     learning

    ,

     and

     other

     emerging

     technologies

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     As

     AI

     becomes

     more

     accurate

     and

     accessible

    ,

     it

     could

     have

     a

     significant

     impact

     on

     healthcare

    .

     For

     example

    ,

     AI

    -powered

     diagnostic

     tools

     could

     help

     doctors

     identify

     diseases

     earlier

    ,

     predict

     patient

     outcomes

    ,

     and

     personalize

     treatment

     plans

    .
    


    2

    .

     Development

     of

     AI

     for

     everyday

     tasks

    :

     AI

     is

     already

     being

     used

     in

     many

     applications

    ,

     including

     self

    -driving

     cars

    ,

     virtual

     assistants

    ,

     and

     chat

    bots

    .

     As

     these

     technologies

     become

     more

     sophisticated

    ,

     we

     may

     see

     even

     more

     widespread

    



```python
llm.shutdown()
```
