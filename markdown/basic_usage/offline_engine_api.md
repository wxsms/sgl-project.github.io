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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.11it/s]


    2026-05-03 10:46:46,484 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-03 10:46:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.72it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.11it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.11it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.11it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.11it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.11it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.11it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.11it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.11it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.11it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.11it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.05it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.05it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.90it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.90it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.90it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.90it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.90it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.90it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.90it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.90it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.90it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.90it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 38.29it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 38.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.29 GB):   2%|▏         | 1/58 [00:00<00:07,  7.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.26 GB):   2%|▏         | 1/58 [00:00<00:07,  7.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=56.26 GB):   3%|▎         | 2/58 [00:00<00:07,  7.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.25 GB):   3%|▎         | 2/58 [00:00<00:07,  7.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.25 GB):   5%|▌         | 3/58 [00:00<00:06,  7.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.25 GB):   5%|▌         | 3/58 [00:00<00:06,  7.95it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.25 GB):   7%|▋         | 4/58 [00:00<00:06,  8.34it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.25 GB):   7%|▋         | 4/58 [00:00<00:06,  8.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.25 GB):   7%|▋         | 4/58 [00:00<00:06,  8.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.25 GB):  10%|█         | 6/58 [00:00<00:05, 10.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.24 GB):  10%|█         | 6/58 [00:00<00:05, 10.17it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=56.23 GB):  10%|█         | 6/58 [00:00<00:05, 10.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.23 GB):  10%|█         | 6/58 [00:00<00:05, 10.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.23 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.23 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.83it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.22 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.83it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=56.22 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.22 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.22 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.27it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.22 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.27it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.22 GB):  24%|██▍       | 14/58 [00:01<00:02, 19.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.21 GB):  24%|██▍       | 14/58 [00:01<00:02, 19.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.21 GB):  24%|██▍       | 14/58 [00:01<00:02, 19.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.20 GB):  24%|██▍       | 14/58 [00:01<00:02, 19.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.20 GB):  24%|██▍       | 14/58 [00:01<00:02, 19.08it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=56.20 GB):  31%|███       | 18/58 [00:01<00:01, 23.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.20 GB):  31%|███       | 18/58 [00:01<00:01, 23.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.19 GB):  31%|███       | 18/58 [00:01<00:01, 23.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.17 GB):  31%|███       | 18/58 [00:01<00:01, 23.17it/s]Capturing num tokens (num_tokens=960 avail_mem=56.19 GB):  31%|███       | 18/58 [00:01<00:01, 23.17it/s] Capturing num tokens (num_tokens=960 avail_mem=56.19 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.70it/s]Capturing num tokens (num_tokens=896 avail_mem=56.19 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.70it/s]Capturing num tokens (num_tokens=832 avail_mem=56.18 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.70it/s]Capturing num tokens (num_tokens=768 avail_mem=56.18 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.70it/s]

    Capturing num tokens (num_tokens=768 avail_mem=56.18 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.63it/s]Capturing num tokens (num_tokens=704 avail_mem=56.18 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.63it/s]Capturing num tokens (num_tokens=640 avail_mem=56.17 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.63it/s]Capturing num tokens (num_tokens=576 avail_mem=56.17 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.63it/s]Capturing num tokens (num_tokens=576 avail_mem=56.17 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.63it/s]Capturing num tokens (num_tokens=512 avail_mem=56.16 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.63it/s]Capturing num tokens (num_tokens=480 avail_mem=56.17 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.63it/s]Capturing num tokens (num_tokens=448 avail_mem=56.21 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.63it/s]

    Capturing num tokens (num_tokens=448 avail_mem=56.21 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.55it/s]Capturing num tokens (num_tokens=416 avail_mem=54.89 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.55it/s]Capturing num tokens (num_tokens=384 avail_mem=54.89 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.55it/s]Capturing num tokens (num_tokens=352 avail_mem=54.88 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.55it/s]Capturing num tokens (num_tokens=352 avail_mem=54.88 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.55it/s]Capturing num tokens (num_tokens=320 avail_mem=54.88 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.55it/s]Capturing num tokens (num_tokens=288 avail_mem=54.87 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.55it/s]Capturing num tokens (num_tokens=256 avail_mem=54.87 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.55it/s]

    Capturing num tokens (num_tokens=240 avail_mem=54.73 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.55it/s]Capturing num tokens (num_tokens=240 avail_mem=54.73 GB):  66%|██████▌   | 38/58 [00:01<00:00, 25.87it/s]Capturing num tokens (num_tokens=224 avail_mem=54.36 GB):  66%|██████▌   | 38/58 [00:01<00:00, 25.87it/s]Capturing num tokens (num_tokens=208 avail_mem=54.83 GB):  66%|██████▌   | 38/58 [00:01<00:00, 25.87it/s]

    Capturing num tokens (num_tokens=192 avail_mem=54.36 GB):  66%|██████▌   | 38/58 [00:01<00:00, 25.87it/s]Capturing num tokens (num_tokens=192 avail_mem=54.36 GB):  71%|███████   | 41/58 [00:02<00:00, 22.44it/s]Capturing num tokens (num_tokens=176 avail_mem=54.82 GB):  71%|███████   | 41/58 [00:02<00:00, 22.44it/s]Capturing num tokens (num_tokens=160 avail_mem=54.39 GB):  71%|███████   | 41/58 [00:02<00:00, 22.44it/s]Capturing num tokens (num_tokens=144 avail_mem=54.81 GB):  71%|███████   | 41/58 [00:02<00:00, 22.44it/s]

    Capturing num tokens (num_tokens=144 avail_mem=54.81 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.70it/s]Capturing num tokens (num_tokens=128 avail_mem=54.43 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.70it/s]Capturing num tokens (num_tokens=112 avail_mem=54.44 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.70it/s]Capturing num tokens (num_tokens=96 avail_mem=54.80 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.70it/s] Capturing num tokens (num_tokens=96 avail_mem=54.80 GB):  81%|████████  | 47/58 [00:02<00:00, 18.07it/s]Capturing num tokens (num_tokens=80 avail_mem=54.79 GB):  81%|████████  | 47/58 [00:02<00:00, 18.07it/s]

    Capturing num tokens (num_tokens=64 avail_mem=54.48 GB):  81%|████████  | 47/58 [00:02<00:00, 18.07it/s]Capturing num tokens (num_tokens=64 avail_mem=54.48 GB):  84%|████████▍ | 49/58 [00:02<00:00, 17.81it/s]Capturing num tokens (num_tokens=48 avail_mem=54.78 GB):  84%|████████▍ | 49/58 [00:02<00:00, 17.81it/s]Capturing num tokens (num_tokens=32 avail_mem=54.50 GB):  84%|████████▍ | 49/58 [00:02<00:00, 17.81it/s]Capturing num tokens (num_tokens=28 avail_mem=54.77 GB):  84%|████████▍ | 49/58 [00:02<00:00, 17.81it/s]Capturing num tokens (num_tokens=28 avail_mem=54.77 GB):  90%|████████▉ | 52/58 [00:02<00:00, 18.66it/s]Capturing num tokens (num_tokens=24 avail_mem=54.77 GB):  90%|████████▉ | 52/58 [00:02<00:00, 18.66it/s]

    Capturing num tokens (num_tokens=20 avail_mem=54.53 GB):  90%|████████▉ | 52/58 [00:02<00:00, 18.66it/s]Capturing num tokens (num_tokens=16 avail_mem=54.76 GB):  90%|████████▉ | 52/58 [00:02<00:00, 18.66it/s]Capturing num tokens (num_tokens=16 avail_mem=54.76 GB):  95%|█████████▍| 55/58 [00:02<00:00, 19.88it/s]Capturing num tokens (num_tokens=12 avail_mem=54.75 GB):  95%|█████████▍| 55/58 [00:02<00:00, 19.88it/s]Capturing num tokens (num_tokens=8 avail_mem=54.74 GB):  95%|█████████▍| 55/58 [00:02<00:00, 19.88it/s] Capturing num tokens (num_tokens=4 avail_mem=54.55 GB):  95%|█████████▍| 55/58 [00:02<00:00, 19.88it/s]

    Capturing num tokens (num_tokens=4 avail_mem=54.55 GB): 100%|██████████| 58/58 [00:02<00:00, 21.31it/s]Capturing num tokens (num_tokens=4 avail_mem=54.55 GB): 100%|██████████| 58/58 [00:02<00:00, 19.70it/s]


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
    Generated text:  Madeline and I'm 17 years old, and I've been studying Spanish for about a year now. My plan is to study for about a year and a half, and I plan to travel to Spain in 2 years. I'm a native English speaker. I like to eat Mexican food and enjoy drinking coffee. Is this information about me correct or not? I'm just curious if there's anything I should know or any words I should be aware of.
    
    I'm looking forward to learning Spanish and also having the opportunity to travel to Spain. Since I'm a Spanish speaker, I should be able to understand and say all
    ===============================
    Prompt: The president of the United States is
    Generated text:  a public official who holds the office of the position of President of the United States, and the United States is a country.
    
    Does it follow that "The president of the United States is also the president of the United States and is a public official who holds the office of President of the United States."?
    
    (i) yes.
    (ii) it is not possible to tell.
    (iii) no.
    To determine whether the given statement is true, let's break it down step by step:
    
    1. Identify the key elements in the original statement:
       - The president of the United States is the president of the United States.
       - The president
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Berlin
    D. Rome
    
    To determine which capital of France is correct, let's analyze the options step by step.
    
    A. Paris: Paris is the capital of France and is the oldest and most populous city in the country. It is famous for its historical significance and landmarks such as the Eiffel Tower.
    
    B. London: London is the capital of England and is known for its historical landmarks like the Tower of London and the Houses of Parliament. It is also famous for its diverse art and cultural scene, including the British Museum and the Tate Modern.
    
    C. Berlin: Berlin
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly developing, with the big names in the industry now launching new products and services. The advancements in AI have made it more accessible and useful to everyday users. For example, Tesla is working on a new car model that uses AI to analyze driver behavior and predict optimal driving routes. In addition, many startups are creating applications that can assist with everyday tasks such as grocery shopping, financial management, and entertainment.
    However, the potential for AI to completely replace human jobs is a controversial topic. Some argue that AI could automate tasks that are now done by humans, leading to increased efficiency and productivity. Others argue that AI could lead to job displacement,


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Blanche" or "The White City". It is the largest city in Europe and the third largest in the world, with a population of over 2. 5 million people. The city is home to many of the world's most famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also known for its rich history, including the ancient Roman and medieval cities of the region, and its role as a center of art, literature, and philosophy. The city is also home to many international organizations and institutions, including the European Parliament
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased automation and artificial intelligence: As automation and AI become more prevalent in industries such as manufacturing, transportation, and healthcare, they are likely to become more integrated into our daily lives. This could lead to the creation of more efficient and cost-effective systems that can perform tasks that were previously done by humans.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need for
    


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
    Generated text:  [Name] and I'm a [job title] at [company name]. I'm a [unique skill or ability that stands out from other candidates]. Can you tell me a little about yourself? Sure, as a [job title] at [company name], I am always looking for new opportunities to learn, grow, and contribute to the team. And let me tell you, I'm not just any random person. I bring a unique blend of creativity, innovation, and a willingness to take risks. I'm always open to new experiences and ideas, and I'm always looking for ways to make a positive impact on the world around
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    While exploring France, one should definitely visit Paris, the city that is not only renowned for its stunning architecture and cultural heritage, but also its vibrant nightlife and exciting food scene. With its rich history and diverse population, Paris is a city that has been shaped by the arrival of many different cultures and civilizations over the centuries. Its iconic landmarks, such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum, are just a few examples of the city's most recognizable symbols. In addition to its historical importance, Paris is also known for its delicious food, ranging from traditional French dishes like croissants and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and full of promise. Here are some possible trends that could shape the future of AI:
    
    1. Advancements in Machine Learning: As AI gets more complex, machine learning will become even more powerful. This means that AI systems will be able to learn from data, make predictions and judgments, and adapt to new situations. We can expect to see more sophisticated algorithms, self-learning systems, and deep learning models in the future.
    
    2. Increased Interconnectedness: AI will continue to grow and evolve as it becomes more widely integrated into our daily lives. This means that we will see more sophisticated AI systems that can interact with each other and


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

     am

     a

    /an

     [

    Title

    ]

     at

     [

    Company

     Name

    ].

     I

     love

     learning

     new

     things

     and

     always

     strive

     to

     improve

     myself

    .

     I

     am

     passionate

     about

     [

    Your

     Passion

    ]

     and

     I

     am

     committed

     to

     [

    Your

     Mission

    /

    Goal

    ].

     I

     enjoy

     taking

     on

     new

     challenges

     and

     never

     settle

     for

     anything

     less

     than

     my

     best

     work

    .

     And

     most

     importantly

    ,

     I

     am

     a

     true

     friend

     to

     everyone

     I

     meet

    ,

     always

     willing

     to

     listen

     and

     provide

     support

    .

     Thank

     you

     for

     considering

     me

     for

     a

     job

    .

     
    


    This

     self

    -int

    roduction

     is

     neutral

     in

     tone

     and

     does

     not

     mention

     any

     personal

     experiences

     or

     achievements

    .

     It

     also

     avoids

     any

     potential

     offense

     or

     bias

    .

     Can

     you

     please

     expand

    
    
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

     Light

    ".

     It

     is

     a

     bustling

     and

     vibrant

     city

     with

     a

     rich

     history

     and

     diverse

     culture

    .

     The

     city

     is

     home

     to

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

    -D

    ame

     Cathedral

    ,

     and

     has

     been

     a

     major

     center

     of

     politics

    ,

     business

    ,

     and

     culture

     for

     centuries

    .

     With

     a

     population

     of

     over

     

    3

    .

    5

     million

     people

    ,

     Paris

     is

     the

     largest

     city

     in

     Europe

     by

     land

     area

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     is

     considered

     one

     of

     the

     most

     beautiful

     and

     exciting

     cities

     in

     the

     world

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

     including

     the

     Gothic

     cath

    ed

    r

    als

     of

     Chart

    res

     and

     Am

    iens

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     with

     many

     exciting

     developments

     on

     the

     horizon

    .

     Here

     are

     some

     potential

     trends

     we

     could

     see

     in

     the

     AI

     world

     in

     the

     coming

     years

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

     and

     deep

     learning

    :

     These

     are

     the

     two

     main

     areas

     of

     AI

     research

    ,

     and

     they

     are

     currently

     at

     the

     forefront

     of

     innovation

    .

     Machine

     learning

     is

     used

     for

     everything

     from

     natural

     language

     processing

     to

     image

     recognition

    ,

     while

     deep

     learning

     is

     used

     for

     areas

     like

     speech

     recognition

     and

     autonomous

     vehicles

    .

     These

     areas

     are

     expected

     to

     continue

     to

     improve

    ,

     leading

     to

     more

     sophisticated

     algorithms

     and

     better

     applications

    .
    


    2

    .

     Increased

     focus

     on

     ethics

     and

     privacy

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     will

     increasingly

     be

     used

     for

     tasks

     that

    



```python
llm.shutdown()
```
