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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.03it/s]


    2026-04-10 08:25:27,952 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 08:25:27] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  4.92it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  4.92it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:09,  4.92it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:09,  4.92it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:09,  4.92it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:03<00:09,  4.92it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:03<00:09,  4.92it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:04,  9.56it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:04,  9.56it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:04,  9.56it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:04,  9.56it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:04,  9.56it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:03<00:04,  9.56it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 13.67it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 19.39it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 19.39it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 19.39it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 19.39it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 19.39it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 19.39it/s]

    Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 19.39it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 25.30it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 25.30it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 25.30it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 25.30it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 25.30it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 25.30it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 29.38it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 29.38it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 29.38it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 29.38it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 29.38it/s]

    Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 29.38it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 32.58it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 34.98it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 34.98it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 34.98it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 34.98it/s]

    Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 34.98it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 34.98it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 34.98it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 40.22it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 40.22it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 40.22it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 40.22it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 40.22it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 40.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   2%|▏         | 1/58 [00:00<00:05,  9.88it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:05,  9.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:05,  9.88it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.76 GB):   5%|▌         | 3/58 [00:00<00:05, 10.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   9%|▊         | 5/58 [00:00<00:04, 12.61it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:04, 12.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:04, 12.61it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:04, 12.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.42it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=118.75 GB):  17%|█▋        | 10/58 [00:00<00:04, 11.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  17%|█▋        | 10/58 [00:00<00:04, 11.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  17%|█▋        | 10/58 [00:00<00:04, 11.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  17%|█▋        | 10/58 [00:00<00:04, 11.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  22%|██▏       | 13/58 [00:00<00:03, 14.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  22%|██▏       | 13/58 [00:00<00:03, 14.22it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  22%|██▏       | 13/58 [00:01<00:03, 14.22it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=118.73 GB):  22%|██▏       | 13/58 [00:01<00:03, 14.22it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.73 GB):  28%|██▊       | 16/58 [00:01<00:02, 16.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  28%|██▊       | 16/58 [00:01<00:02, 16.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  28%|██▊       | 16/58 [00:01<00:02, 16.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.72 GB):  28%|██▊       | 16/58 [00:01<00:02, 16.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.72 GB):  33%|███▎      | 19/58 [00:01<00:01, 19.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  33%|███▎      | 19/58 [00:01<00:01, 19.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  33%|███▎      | 19/58 [00:01<00:01, 19.72it/s]

    Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  33%|███▎      | 19/58 [00:01<00:01, 19.72it/s] Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  33%|███▎      | 19/58 [00:01<00:01, 19.72it/s]Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.55it/s]Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.55it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.55it/s]Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.55it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.55it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.55it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.08it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.08it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.08it/s]

    Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.08it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.08it/s]Capturing num tokens (num_tokens=384 avail_mem=118.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.08it/s]Capturing num tokens (num_tokens=384 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=224 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.31it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.98it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.98it/s]

    Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.98it/s]Capturing num tokens (num_tokens=128 avail_mem=118.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 30.98it/s]Capturing num tokens (num_tokens=112 avail_mem=118.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 30.98it/s]Capturing num tokens (num_tokens=112 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.60it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.60it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.60it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.60it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:02<00:00, 17.95it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:02<00:00, 17.95it/s]

    Capturing num tokens (num_tokens=32 avail_mem=118.63 GB):  84%|████████▍ | 49/58 [00:02<00:00, 17.95it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  84%|████████▍ | 49/58 [00:02<00:00, 17.95it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:02<00:00, 18.50it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:02<00:00, 18.50it/s]Capturing num tokens (num_tokens=20 avail_mem=118.62 GB):  90%|████████▉ | 52/58 [00:02<00:00, 18.50it/s]Capturing num tokens (num_tokens=16 avail_mem=118.62 GB):  90%|████████▉ | 52/58 [00:02<00:00, 18.50it/s]Capturing num tokens (num_tokens=16 avail_mem=118.62 GB):  95%|█████████▍| 55/58 [00:02<00:00, 20.22it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  95%|█████████▍| 55/58 [00:02<00:00, 20.22it/s]

    Capturing num tokens (num_tokens=8 avail_mem=118.61 GB):  95%|█████████▍| 55/58 [00:02<00:00, 20.22it/s] Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  95%|█████████▍| 55/58 [00:02<00:00, 20.22it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 21.08it/s]


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
    Generated text:  Tanyra Hill, and I'm 16 years old. I'm not your typical 16 year old. I was born with a physical condition called cleft palate. I was born without the roof of my mouth. I didn't have a palate to breathe. I never have a palate. I'm also missing my ear. In 2005, I had a cleft palate repair surgery. I had a cleft palate repair surgery on July 16, 2005. My surgery was successful. I got to go home. I was 2 years old. I was a baby.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a public servant in charge of a broad range of roles in the government of the country. Here are the positions and responsibilities of the president of the United States:  "It's important to know who the president is and what their positions are in order to help them lead effectively and responsibly. "
    The president of the United States is the head of state and government of the United States, and the president is the highest elected official of the United States. The president is the commander-in-chief of the armed forces of the United States and the head of the executive branch of the United States government. It's important to know who the president is and
    ===============================
    Prompt: The capital of France is
    Generated text:  the most important city of the French Republic, located on the right bank of the Seine. It is the capital of the Île-de-France region and has a total of 53 districts. The current mayor of Paris is Claude Leca.
    Is there an answer to this question (If it cannot be answered, return "Unanswerable"). Answer:
    
    Is the capital of France located on the right bank of the Seine? Yes, the capital of France is located on the right bank of the Seine. This statement is unanswerable as there is no specific information provided in the given text that addresses the location of the
    ===============================
    Prompt: The future of AI is
    Generated text:  to become increasingly intelligent. AI, or Artificial Intelligence, is the ability of machines to perform tasks that traditionally require human intelligence, including reasoning, learning, problem solving, and decision making. Some of the machines that can perform these tasks are facial recognition software, self-driving cars, and virtual assistants that assist people with tasks, such as answering questions, making appointments, and booking travel. With the development of AI, we can look at the areas that are going to be more diverse and intelligent in the future. Some of the areas that are likely to see more development in the future include healthcare, energy, transportation, and education. These are just


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major economic and political center in Europe. It is also known for its fashion industry and is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a popular tourist destination and is considered one of the most beautiful cities in the world. It is also home to the French Riv
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more natural and intuitive interactions between humans and machines.
    
    2. Enhanced machine learning capabilities: AI is likely to become even more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective applications of AI in various fields.
    
    3. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be increased pressure
    


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
    Generated text:  [Your Name]. I'm a [Your Profession] and I'm currently [Your Current Position]. I'm always looking to learn new things and improve myself. I enjoy problem-solving and thrives in a fast-paced environment. I'm a great communicator and enjoy collaborating with others to achieve our goals. If you'd like to meet me, please feel free to introduce yourself to me. Do you have any particular interests or hobbies that you enjoy? I'm always looking for new opportunities to engage with new people and broaden my horizons. Thank you for considering me for a job interview. Good luck! What are some ways that a job candidate
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. It is the third most populous city in the world and is a major economic, cultural, and political center. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also famous for its fashion industry, particularly the couture dresses made by high-end designers like Yves Saint Laurent and Christian Dior. Paris is home to many museums and art galleries, as well as a vibrant arts scene. The city is also home to many cultural events and festivals throughout the year. The weather in Paris can vary greatly from
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be a rapidly evolving area of research and development, driven by advancements in machine learning, computer hardware, and big data analysis. Here are some possible trends we can expect to see in the future of AI:
    
    1. Increased Personalization: One of the key trends we can expect to see in AI in the future is increased personalization. AI will be able to learn from a vast amount of data to provide more accurate and relevant recommendations and suggestions to users. This will allow users to find products and services that are most likely to meet their needs and preferences.
    
    2. AI will become more connected: The next big trend is the increasing


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

    'm

     [

    Your

     Age

    ]

     years

     old

    .

     I

     come

     from

     [

    Your

     Country

    ]

     and

     have

     always

     been

     interested

     in

     [

    Your

     Hobby

    /

    Interest

    ].

     I

     enjoy

     [

    Your

     Hobby

    /

    Interest

    ]

     because

     it

     helps

     me

     [

    Reason

     for

     Hobby

    /

    Interest

    ].

     I

     enjoy

     [

    Your

     Hobby

    /

    Interest

    ]

     because

     it

     [

    Reason

     for

     Hobby

    /

    Interest

    ].

     I

     have

     always

     been

     passionate

     about

     [

    Your

     Hobby

    /

    Interest

    ]

     because

     [

    Reason

     for

     Hobby

    /

    Interest

    ].

     I

     am

     always

     eager

     to

     learn

     and

     share

     my

     knowledge

     with

     others

    .

     What

    's

     your

     favorite

     hobby

    /

    interest

    ?

     What

    's

     your

     greatest

     accomplishment

    ?

     What

    's

     your

     biggest

     challenge

    ?

     What

    's

     the

     best

     advice

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    A

    .

     True

    


    B

    .

     False

    
    


    The

     answer

     is

     A

    .

     True

    .

     Paris

     is

     the

     capital

     city

     of

     France

     and

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

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     a

     major

     hub

     for

     cultural

     and

     business

     activities

    .

     According

     to

     the

     U

    .S

    .

     State

     of

     the

     City

     report

    ,

     Paris

     has

     a

     population

     of

     over

     

    1

    .

    3

     million

     people

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     area

     and

     is

     home

     to

     many

     museums

    ,

     theaters

    ,

     and

     other

     cultural

     institutions

    .

     Lastly

    ,

     Paris

     is

     known

     for

     its

     romantic

     and

     enchant

    ing

     atmosphere

    ,

     and

     it

     is

     a

     popular

     destination

     for

     tourists

     from

     around

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     advancements

     in

     both

     hardware

     and

     software

     technologies

    .

     One

     of

     the

     most

     exciting

     areas

     of

     AI

     development

     is

     the

     development

     of

     machine

     learning

     and

     artificial

     intelligence

     systems

     that

     can

     learn

     and

     adapt

     to

     new

     situations

    ,

     without

     being

     explicitly

     programmed

    .

     This

     is

     called

     "

    art

    ificial

     general

     intelligence

    "

     (

    AG

    I

    ),

     and

     it

     is

     currently

     the

     most

     challenging

     goal

     in

     AI

     research

    .


    As

     of

     now

    ,

     some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :


    1

    .

     Increased

     reliance

     on

     AI

     in

     areas

     like

     healthcare

     and

     finance

    ,

     where

     traditional

     methods

     may

     be

     slow

     or

     in

    feas

    ible

    .


    2

    .

     The

     development

     of

     AI

     that

     can

     understand

     and

     interpret

     human

     emotions

    ,

     thoughts

    ,

     and

     language

    



```python
llm.shutdown()
```
