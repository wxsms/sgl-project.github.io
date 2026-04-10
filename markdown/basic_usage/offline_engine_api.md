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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.98it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.96it/s]


    2026-04-10 08:46:12,002 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 08:46:12] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.10it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:25,  2.10it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:11,  4.42it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:11,  4.42it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:03<00:11,  4.42it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:03<00:11,  4.42it/s]

    Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:03<00:11,  4.42it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  8.02it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  8.02it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  8.02it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  8.02it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:05,  8.02it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:05,  8.02it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 13.22it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 13.22it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 13.22it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 13.22it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:03, 13.22it/s]

    Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:03<00:03, 13.22it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:03<00:03, 13.22it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:03<00:03, 13.22it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:01, 21.66it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:01, 21.66it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:01, 21.66it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:01, 21.66it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:01, 21.66it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:01, 21.66it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:03<00:01, 21.66it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:03<00:01, 21.66it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:00, 29.86it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:00, 29.86it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:00, 29.86it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:00, 29.86it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:00, 29.86it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:00, 29.86it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:00, 29.86it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:00, 29.86it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 36.49it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 36.49it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 36.49it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 36.49it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 36.49it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 36.49it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 36.49it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 41.08it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 41.08it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 41.08it/s]

    Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 41.08it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 41.08it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 41.08it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 41.08it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 41.96it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 41.96it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 41.96it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 41.96it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 41.96it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 41.96it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 41.96it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 45.96it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 45.96it/s]

    Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 45.96it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 45.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   2%|▏         | 1/58 [00:00<00:06,  9.35it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:06,  9.35it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.76 GB):   2%|▏         | 1/58 [00:00<00:06,  9.35it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=118.76 GB):   5%|▌         | 3/58 [00:00<00:04, 11.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.76 GB):   5%|▌         | 3/58 [00:00<00:04, 11.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.76 GB):   5%|▌         | 3/58 [00:00<00:04, 11.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   5%|▌         | 3/58 [00:00<00:04, 11.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 16.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 16.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 16.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 16.97it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=118.75 GB):  10%|█         | 6/58 [00:00<00:03, 16.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.75 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.16it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.16it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.16it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.12it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=118.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.21 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.03 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.12it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=118.03 GB):  28%|██▊       | 16/58 [00:00<00:02, 14.50it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.02 GB):  28%|██▊       | 16/58 [00:00<00:02, 14.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.02 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.02 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.02 GB):  33%|███▎      | 19/58 [00:01<00:02, 16.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.01 GB):  33%|███▎      | 19/58 [00:01<00:02, 16.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=117.99 GB):  33%|███▎      | 19/58 [00:01<00:02, 16.61it/s]Capturing num tokens (num_tokens=960 avail_mem=118.01 GB):  33%|███▎      | 19/58 [00:01<00:02, 16.61it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=118.01 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.19it/s]Capturing num tokens (num_tokens=896 avail_mem=118.00 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.19it/s]Capturing num tokens (num_tokens=832 avail_mem=118.00 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.19it/s]Capturing num tokens (num_tokens=768 avail_mem=118.00 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.19it/s]Capturing num tokens (num_tokens=704 avail_mem=117.99 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.19it/s]Capturing num tokens (num_tokens=704 avail_mem=117.99 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.70it/s]Capturing num tokens (num_tokens=640 avail_mem=117.99 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.70it/s]Capturing num tokens (num_tokens=576 avail_mem=117.99 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.70it/s]Capturing num tokens (num_tokens=512 avail_mem=117.98 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.70it/s]

    Capturing num tokens (num_tokens=480 avail_mem=117.99 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.70it/s]Capturing num tokens (num_tokens=480 avail_mem=117.99 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.80it/s]Capturing num tokens (num_tokens=448 avail_mem=117.99 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.80it/s]Capturing num tokens (num_tokens=416 avail_mem=117.99 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.80it/s]Capturing num tokens (num_tokens=384 avail_mem=117.99 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.80it/s]Capturing num tokens (num_tokens=352 avail_mem=117.98 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.80it/s]Capturing num tokens (num_tokens=352 avail_mem=117.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.78it/s]Capturing num tokens (num_tokens=320 avail_mem=117.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.78it/s]Capturing num tokens (num_tokens=288 avail_mem=117.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.78it/s]

    Capturing num tokens (num_tokens=256 avail_mem=117.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.78it/s]Capturing num tokens (num_tokens=240 avail_mem=117.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.78it/s]

    Capturing num tokens (num_tokens=240 avail_mem=117.97 GB):  66%|██████▌   | 38/58 [00:01<00:00, 21.90it/s]Capturing num tokens (num_tokens=224 avail_mem=118.86 GB):  66%|██████▌   | 38/58 [00:01<00:00, 21.90it/s]Capturing num tokens (num_tokens=208 avail_mem=118.85 GB):  66%|██████▌   | 38/58 [00:01<00:00, 21.90it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.90it/s]

    Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  71%|███████   | 41/58 [00:02<00:00, 19.10it/s]Capturing num tokens (num_tokens=176 avail_mem=137.28 GB):  71%|███████   | 41/58 [00:02<00:00, 19.10it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  71%|███████   | 41/58 [00:02<00:00, 19.10it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  71%|███████   | 41/58 [00:02<00:00, 19.10it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.22it/s]Capturing num tokens (num_tokens=128 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.22it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.22it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.22it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.22it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.22it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.35it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.35it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.35it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.35it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.35it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.44it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.44it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.44it/s]

    Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.44it/s]Capturing num tokens (num_tokens=8 avail_mem=137.23 GB):  91%|█████████▏| 53/58 [00:02<00:00, 27.44it/s] Capturing num tokens (num_tokens=8 avail_mem=137.23 GB):  98%|█████████▊| 57/58 [00:02<00:00, 29.86it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  98%|█████████▊| 57/58 [00:02<00:00, 29.86it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:02<00:00, 22.53it/s]


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
    Generated text:  Daniela. My favorite hobby is gardening. I have a total of 50 flowers in my garden. Some are yellow, and some are purple. I have twice as many yellow flowers as purple flowers. How many yellow flowers are in my garden? Let's denote the number of purple flowers as \( p \). According to the information given, the number of yellow flowers is twice the number of purple flowers, so the number of yellow flowers is \( 2p \).
    
    The total number of flowers is the sum of the yellow and purple flowers, which is given as 50. So we can write the equation:
    
    \[ p
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected office. No one is elected president of the United States. Instead, the president is elected to a fixed term of four years. Most candidates run on the promise of legislation to improve the quality of life in the United States.
    
    Who is the president of the United States? (If the question is unanswerable, say "Unanswerable"). Unanswerable.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Which country has the same capital as Paris?
    A. England
    B. Switzerland
    C. Germany
    D. Italy
    The correct answer is C. Germany.
    Germany is the country that has the same capital as Paris. The capital of Germany is Berlin. France's capital is Paris, and Switzerland's capital is Bern.
    To solve this problem, we need to identify the countries that share the same capital as Paris. This can be determined by comparing the capitals of these countries. For example, the capital of England is London, while the capital of Switzerland is Bern. Similarly, the capital of Germany is Berlin.
    Therefore, the
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to be AI in the cloud, rather than AI in the cloud. As we move towards a future where data is processed on the edge rather than in the cloud, AI must be cloud-aware. This means that all data at the edge must be processed on the cloud in order to be processed and used to create insights.
    The AI in the cloud will likely require a different set of data processing models than the AI in the cloud. For example, the AI in the cloud will likely need to work with large amounts of data that are not formatted in the same way as the AI in the cloud. This can make it more difficult for the


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also a major economic center and a major player in international politics. The city is known for its fashion industry, art scene, and food culture. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that has played a significant role in French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to help diagnose and treat diseases, predict patient outcomes, and improve patient care. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the future.
    
    2. Increased use of AI in finance: AI is already being used in finance to help with fraud detection, risk management, and portfolio optimization. As AI
    


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
    Generated text:  [Name] and I'm a [Type]! I am passionate about [Purpose of Interest] and always strive to [Hobby/Activity]. I am [Age] years old and I have [Number of Years of Experience/Professional Experience] years in this field. My journey in [Field of Interest] is [Title or Achievement], where [Name of Organization/Company] serves as the [Role]. I am constantly learning and growing, always striving to [Hobby/Activity] and [Purpose of Interest]. As a [Type], I am a [Example of Personality Type], known for [Example of Personality Trait]. My
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city where the French Revolution began and continues to this day.
    
    The statement can be summarized as follows:
    
    Paris, the capital of France, is the city where the French Revolution began and continues to this day. 
    
    This statement encapsulates the core facts about Paris in French, indicating where the event that marked the beginning of the French Revolution happened and continuing to be a significant event in French and global history. 
    
    To summarize the statement in a single sentence:
    "Paris, the capital of France, is the location where the French Revolution began and continues to be remembered as an important historical event in France and beyond." 
    
    This concise statement
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising and exciting, with many possible trends that could shape the technology in the coming decades. Here are a few of the most likely future trends in AI:
    
    1. Increased accuracy and efficiency: As AI gets better at recognizing patterns and making decisions, we're likely to see even more accurate and efficient AI systems. This could lead to a variety of applications, from self-driving cars to personalized medicine to fraud detection systems.
    
    2. AI in healthcare: AI has the potential to revolutionize the healthcare industry by improving diagnosis accuracy, reducing errors, and making treatments more personalized. This could lead to new treatments for diseases such as cancer, HIV/AIDS,


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

     who

     has

     been

     [

    time

    ]

     years

     of

     experience

     in

     this

     field

    .

     I

     have

     a

     genuine

     passion

     for

     [

    job

     title

    ]

     and

     I

     enjoy

     [

    activities

     or

     hobbies

    ]

     which

     make

     me

     unique

    .

     I

     am

     always

     ready

     to

     learn

     and

     help

     anyone

     who

     comes

     to

     me

     with

     a

     question

    .

     I

     am

     a

     kind

     and

     approach

    able

     person

     who

     is

     always ready

     to

     assist

     and

     support

     you

    .

     I

     am

     passionate about

     using

     my

     skills

     and

     experience

     to

     make

     a

     positive

     impact

     on

     the

     world

     and

     to

     help

     others

    .

     I

     am

     confident

     in

     my

     abilities

     and

     will

     do

     my

     best

     to

     help

     you

     achieve

     your

     goals

    .

     If

     you

     need

     anything

     today

    ,

     please

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     center

     of

     the

     country

     on

     the

     River

     Se

    ine

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     iconic

     architecture

    ,

     and

     vibrant

     cultural

     scene

    .

     Paris

     is

     the

     world

    's

     most

     populous

     city

     and

     has

     a

     population

     of

     over

     

    1

    8

     million

     people

    .

     The

     city

     is

     also

     home

     to

     numerous

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

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     cultural

     hub

     in

     Europe

    .

     It

     is

     known

     for

     its

     unique

     blend

     of

     French

     traditions

     and

     modern

    ity

    ,

     and

     is

     a

     major

     center

     for

     business

    ,

     finance

    ,

     and

     arts

    .

     The

     city

     has

     been

     recognized

     as

     one

     of

     the

     world

    's

     most

     liv

    able

    
    
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

     progress

     and

     challenges

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

     Advances

     in

     neural

     networks

     and

     machine

     learning

    :

     As

     neural

     networks

     and

     machine

     learning

     become

     more

     sophisticated

    ,

     we

     are

     likely

     to

     see

     more

     advanced

     AI

     systems

     that

     can

     learn

     from

     vast

     amounts

     of

     data

     and

     make

     accurate

     predictions

     and

     decisions

    .

     This

     could

     lead

     to

     breakthrough

    s

     in

     fields

     like

     medicine

    ,

     finance

    ,

     and

     automation

    .
    


    2

    .

     Increased

     reliance

     on

     AI

     in

     various

     industries

    :

     As

     AI

     becomes

     more

     integrated

     into

     various

     industries

    ,

     we

     are

     likely

     to

     see

     more

     widespread

     adoption

     of

     AI

     solutions

    .

     This

     could

     lead

     to

     increased

     efficiency

    ,

     cost

     savings

    ,

     and

     better

     decision

    -making

     for

     businesses

    



```python
llm.shutdown()
```
