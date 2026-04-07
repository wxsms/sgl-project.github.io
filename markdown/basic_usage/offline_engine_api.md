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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.29it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.29it/s]


    2026-04-07 02:11:38,647 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 02:11:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:41,  1.32it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:41,  1.32it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:41,  1.32it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:41,  1.32it/s]

    Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:02<00:41,  1.32it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:13,  3.81it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:13,  3.81it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:05,  8.44it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:05,  8.44it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:05,  8.44it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:05,  8.44it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:05,  8.44it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:03<00:05,  8.44it/s]

    Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:03<00:05,  8.44it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:03<00:05,  8.44it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 14.76it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 14.76it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 14.76it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 14.76it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 14.76it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 14.76it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 14.76it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 14.76it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 21.78it/s]

    Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 21.78it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 28.99it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 28.99it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 28.99it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 28.99it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 28.99it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 28.99it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:03<00:00, 28.99it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 34.34it/s]

    Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 34.34it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 34.34it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 49.10it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 49.10it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 49.10it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 49.10it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=74.46 GB):   2%|▏         | 1/58 [00:00<00:31,  1.80it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.11 GB):   2%|▏         | 1/58 [00:00<00:31,  1.80it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.11 GB):   3%|▎         | 2/58 [00:00<00:17,  3.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.15 GB):   3%|▎         | 2/58 [00:00<00:17,  3.21it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.15 GB):   5%|▌         | 3/58 [00:00<00:11,  4.61it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   5%|▌         | 3/58 [00:00<00:11,  4.61it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   7%|▋         | 4/58 [00:01<00:12,  4.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.16 GB):   7%|▋         | 4/58 [00:01<00:12,  4.28it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:01<00:11,  4.43it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:01<00:11,  4.43it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.17 GB):  10%|█         | 6/58 [00:01<00:16,  3.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.30 GB):  10%|█         | 6/58 [00:01<00:16,  3.16it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=74.30 GB):  12%|█▏        | 7/58 [00:02<00:15,  3.22it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.31 GB):  12%|█▏        | 7/58 [00:02<00:15,  3.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.33 GB):  12%|█▏        | 7/58 [00:02<00:15,  3.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.33 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.21 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.20 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.27it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.31 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.31 GB):  21%|██        | 12/58 [00:02<00:05,  8.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.30 GB):  21%|██        | 12/58 [00:02<00:05,  8.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.27 GB):  21%|██        | 12/58 [00:02<00:05,  8.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.29 GB):  21%|██        | 12/58 [00:02<00:05,  8.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.29 GB):  26%|██▌       | 15/58 [00:02<00:03, 12.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.28 GB):  26%|██▌       | 15/58 [00:02<00:03, 12.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.27 GB):  26%|██▌       | 15/58 [00:02<00:03, 12.17it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=74.25 GB):  26%|██▌       | 15/58 [00:02<00:03, 12.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.26 GB):  26%|██▌       | 15/58 [00:02<00:03, 12.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.26 GB):  33%|███▎      | 19/58 [00:02<00:02, 17.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.25 GB):  33%|███▎      | 19/58 [00:02<00:02, 17.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.23 GB):  33%|███▎      | 19/58 [00:02<00:02, 17.02it/s]Capturing num tokens (num_tokens=960 avail_mem=74.24 GB):  33%|███▎      | 19/58 [00:02<00:02, 17.02it/s] Capturing num tokens (num_tokens=896 avail_mem=74.23 GB):  33%|███▎      | 19/58 [00:02<00:02, 17.02it/s]Capturing num tokens (num_tokens=832 avail_mem=74.23 GB):  33%|███▎      | 19/58 [00:02<00:02, 17.02it/s]Capturing num tokens (num_tokens=832 avail_mem=74.23 GB):  41%|████▏     | 24/58 [00:02<00:01, 22.93it/s]Capturing num tokens (num_tokens=768 avail_mem=74.22 GB):  41%|████▏     | 24/58 [00:02<00:01, 22.93it/s]Capturing num tokens (num_tokens=704 avail_mem=74.21 GB):  41%|████▏     | 24/58 [00:02<00:01, 22.93it/s]

    Capturing num tokens (num_tokens=640 avail_mem=74.22 GB):  41%|████▏     | 24/58 [00:02<00:01, 22.93it/s]Capturing num tokens (num_tokens=576 avail_mem=74.20 GB):  41%|████▏     | 24/58 [00:02<00:01, 22.93it/s]Capturing num tokens (num_tokens=512 avail_mem=74.18 GB):  41%|████▏     | 24/58 [00:02<00:01, 22.93it/s]Capturing num tokens (num_tokens=512 avail_mem=74.18 GB):  50%|█████     | 29/58 [00:02<00:01, 27.98it/s]Capturing num tokens (num_tokens=480 avail_mem=74.20 GB):  50%|█████     | 29/58 [00:02<00:01, 27.98it/s]Capturing num tokens (num_tokens=448 avail_mem=74.19 GB):  50%|█████     | 29/58 [00:02<00:01, 27.98it/s]Capturing num tokens (num_tokens=416 avail_mem=74.18 GB):  50%|█████     | 29/58 [00:02<00:01, 27.98it/s]Capturing num tokens (num_tokens=384 avail_mem=74.18 GB):  50%|█████     | 29/58 [00:02<00:01, 27.98it/s]Capturing num tokens (num_tokens=352 avail_mem=74.17 GB):  50%|█████     | 29/58 [00:02<00:01, 27.98it/s]Capturing num tokens (num_tokens=352 avail_mem=74.17 GB):  59%|█████▊    | 34/58 [00:02<00:00, 31.78it/s]Capturing num tokens (num_tokens=320 avail_mem=74.16 GB):  59%|█████▊    | 34/58 [00:02<00:00, 31.78it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.16 GB):  59%|█████▊    | 34/58 [00:02<00:00, 31.78it/s]Capturing num tokens (num_tokens=256 avail_mem=74.15 GB):  59%|█████▊    | 34/58 [00:02<00:00, 31.78it/s]Capturing num tokens (num_tokens=240 avail_mem=74.17 GB):  59%|█████▊    | 34/58 [00:02<00:00, 31.78it/s]Capturing num tokens (num_tokens=240 avail_mem=74.17 GB):  66%|██████▌   | 38/58 [00:03<00:00, 33.48it/s]Capturing num tokens (num_tokens=224 avail_mem=74.16 GB):  66%|██████▌   | 38/58 [00:03<00:00, 33.48it/s]Capturing num tokens (num_tokens=208 avail_mem=74.16 GB):  66%|██████▌   | 38/58 [00:03<00:00, 33.48it/s]Capturing num tokens (num_tokens=192 avail_mem=74.15 GB):  66%|██████▌   | 38/58 [00:03<00:00, 33.48it/s]Capturing num tokens (num_tokens=176 avail_mem=74.15 GB):  66%|██████▌   | 38/58 [00:03<00:00, 33.48it/s]Capturing num tokens (num_tokens=160 avail_mem=74.14 GB):  66%|██████▌   | 38/58 [00:03<00:00, 33.48it/s]Capturing num tokens (num_tokens=160 avail_mem=74.14 GB):  74%|███████▍  | 43/58 [00:03<00:00, 36.55it/s]Capturing num tokens (num_tokens=144 avail_mem=74.13 GB):  74%|███████▍  | 43/58 [00:03<00:00, 36.55it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.13 GB):  74%|███████▍  | 43/58 [00:03<00:00, 36.55it/s]Capturing num tokens (num_tokens=112 avail_mem=74.12 GB):  74%|███████▍  | 43/58 [00:03<00:00, 36.55it/s]Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  74%|███████▍  | 43/58 [00:03<00:00, 36.55it/s] Capturing num tokens (num_tokens=80 avail_mem=74.11 GB):  74%|███████▍  | 43/58 [00:03<00:00, 36.55it/s]Capturing num tokens (num_tokens=80 avail_mem=74.11 GB):  83%|████████▎ | 48/58 [00:03<00:00, 38.87it/s]Capturing num tokens (num_tokens=64 avail_mem=74.10 GB):  83%|████████▎ | 48/58 [00:03<00:00, 38.87it/s]Capturing num tokens (num_tokens=48 avail_mem=74.10 GB):  83%|████████▎ | 48/58 [00:03<00:00, 38.87it/s]Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  83%|████████▎ | 48/58 [00:03<00:00, 38.87it/s]Capturing num tokens (num_tokens=28 avail_mem=74.08 GB):  83%|████████▎ | 48/58 [00:03<00:00, 38.87it/s]Capturing num tokens (num_tokens=24 avail_mem=74.07 GB):  83%|████████▎ | 48/58 [00:03<00:00, 38.87it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.07 GB):  91%|█████████▏| 53/58 [00:03<00:00, 39.03it/s]Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  91%|█████████▏| 53/58 [00:03<00:00, 39.03it/s]Capturing num tokens (num_tokens=16 avail_mem=74.06 GB):  91%|█████████▏| 53/58 [00:03<00:00, 39.03it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  91%|█████████▏| 53/58 [00:03<00:00, 39.03it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  91%|█████████▏| 53/58 [00:03<00:00, 39.03it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  91%|█████████▏| 53/58 [00:03<00:00, 39.03it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:03<00:00, 40.72it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:03<00:00, 16.74it/s]


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
    Generated text:  Mabel. I am a five-year-old girl. I am from a small village in the countryside. I have a friend, Jim. We are both very kind people. But Jim is younger than me and he is shy. He doesn't like to play with me because he thinks I'm too old. Jim goes to a big school and plays with other kids there. But he doesn't get along with us very well. Mabel and Jim are good friends. Now, write a 600-word composition about what you think of Mabel and Jim and how they are good friends. What are your thoughts on the friendship between
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking federal executive officer. President Obama served two terms from 2009 to 2017. During his term, the United States experienced a 20% increase in the size of the federal government. The president also announced a series of initiatives designed to address economic challenges, including the creation of the Federal Reserve, the Affordable Care Act, and the Dodd-Frank Wall Street Reform and Consumer Protection Act. These initiatives were instrumental in stabilizing the economy and reducing unemployment.
    
    The president also served in several high-ranking positions in the Department of Defense, including the commander of the U.S. 5th Infantry Division
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Rome
    C. Amsterdam
    D. Hong Kong
    Answer:
    
    A
    
    A survey of the average annual interest rate on deposits at the end of 2003 showed that the interest rate on deposits at the end of 2003 was 5.00%, with a standard deviation of 0.02%. Using the standard normal distribution, the approximate percentage of deposits with interest rates within the range 5% to 6% would be ____.
    A. 4.8%
    B. 95%
    C. 16%
    D. 84
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it's coming faster than you might think.
    In this series, we'll be discussing the big ideas in the field of artificial intelligence, with a focus on what it means for people, businesses, and governments. In each episode, we'll explore a different AI-related topic and examine the latest advancements in the field.
    Today, we're looking at the future of AI. In the past, AI has been a technology that's available only in the tech companies. But in the coming years, AI will become more widely used in the everyday world.
    But the biggest challenge in the coming years will be developing the right AI systems that


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to meet you. I'm [name] and I'm here to help you with any questions you may have. I'm a [job title] at [company name], and I'm here to help you with any questions you may have. I'm a [job title] at [company name], and I'm here to help you with any questions you may have. I'm a [job title] at [company name], and I'm here to help you with any
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. Paris is home to many famous museums, including the Louvre and the Musée d'Orsay, as well as the Notre-Dame Cathedral, the Eiffel Tower, and the Champs-Élysées. The city is also known for its fashion industry, with many famous designers and boutiques located in the city center
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends in AI that are expected to shape the future:
    
    1. Increased automation and artificial intelligence: As AI continues to advance, we are likely to see an increase in automation and artificial intelligence in various industries. This could lead to the creation of new jobs, but it could also lead to the creation of new jobs in areas such as data analysis, machine learning, and software development.
    
    2. Improved privacy and security: As AI becomes more advanced, there will be an increased need
    


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
    Generated text:  [Your Name], and I'm a [Your occupation] who works at [Your company or job title]. I'm excited to meet you and learn more about your experiences and skills. Let's have a chat and see where our friendship can take us. What's your name? What's your job? What do you do? How can I help you today? I'm looking forward to our conversation. Let's connect. [Your Name] [Your email address] [Your phone number] [Your LinkedIn or other professional profile URL] [Your social media handles] [Your other contact information] [Your website or website address] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, and it is located on the Seine River in the center of the country. The city is the cultural, economic, and political center of France, and has a rich history dating back to ancient times. Paris is known for its stunning architecture, including the Eiffel Tower and the Louvre Museum. The city also has a thriving arts and culture scene, with numerous galleries and museums, as well as numerous music and theater venues. Paris is a popular tourist destination, and is home to numerous historic and modern landmarks. Overall, Paris is a city that is a true Parisian touchstone, with a rich history and a vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by significant advancements in a number of areas, including:
    
    1. Improved algorithms and machine learning: As technology continues to improve, we can expect to see even more sophisticated algorithms and machine learning systems that can make more accurate predictions and decisions.
    
    2. Increased use of robotics and automation: AI-powered robotics and automation in manufacturing and other industries could lead to increased productivity and efficiency.
    
    3. Enhanced natural language processing: With the increasing amount of data being collected and analyzed, AI systems will be able to understand and interpret natural language much better, leading to increased automation and efficiency.
    
    4. Deep learning and neural networks: As the technology


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

    career

    ]

     at

     [

    company

    /

    organization

    ].

     My

     strength

     lies

     in

     [

    specific

     skill

     or

     attribute

    ]

     and

     my

     experience

     includes

     [

    number

     of

     years

     of

     experience

    ].

     I

     am

     passionate

     about

     [

    why

     you

     are

     passionate

     about

     your

     career

    ].

     I

     am

     a

     [

    career

    ]

     at

     [

    company

    /

    organization

    ],

     and

     I

     believe

     that

     [

    what

     motiv

    ates

     you

     as

     a

     person

     and

     what

     you

     strive

     to

     achieve

     in

     your

     career

    ].

     I

     am

     a

     [

    career

    ]

     at

     [

    company

    /

    organization

    ],

     and

     I

     am

     dedicated

     to

     [

    career

     goals

    ].


    Your

     introduction

     should

     be

     brief

     but

     informative

    ,

     highlighting

     your

     specific

     skills

    ,

     experience

    ,

     and

     passion

     for

     your

     career

    .

     Use

     neutral

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     France

    ,

     hosting

     the

     Lou

    vre

     Museum

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     landmarks

    .

     The

     city

     is

     also

     known

     for

     its

     fashion

    ,

     wine

    ,

     and

     food

     scenes

    .

     Paris

     is

     renowned

     for

     its

     iconic

     landmarks

     and

     historic

     architecture

    ,

     including

     the

     E

    iff

    el

     Tower

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

     vibrant

     nightlife

     and

     music

     scene

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

     offering

     a

     variety

     of

     experiences

     and

     activities

    .

     The

     city

     is

     a

     cultural

     and

     political

     powerhouse

     in

     France

     and

     plays

     a

     significant

     role

     in

     the

     country

    's

     economy

    .

     Paris

     is

     the

     capital

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     constantly

     evolving

     and

     there

     are

     several

     trends

     that

     are

     likely

     to

     shape

     the

     way

     we

     live

     and

     work

     in

     the

     coming

     years

    .

     Some

     of

     the

     most

     likely

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     The

     development

     of

     more

     advanced

     AI

     systems

     that

     can

     learn

     and

     adapt

     to

     human

    -like

     behaviors

     may

     lead

     to

     a

     more

     integrated

     AI

    -human

     interaction

    ,

     where

     AI

     can

     assist

     humans

     in

     decision

    -making

     and

     problem

    -solving

    .
    


    2

    .

     AI

     becoming

     more

     autonomous

    :

     The

     development

     of

     autonomous

     AI

     systems

     that

     can

     operate

     without

     human

     intervention

     may

     become

     more

     prevalent

     in

     the

     future

    ,

     leading

     to

     a

     more

     efficient

     and

     intelligent

     society

    .
    


    3

    .

     AI

     becoming

     more

     versatile

    :

     AI

     systems

     that

     can

     perform

     tasks

    



```python
llm.shutdown()
```
