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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.20it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.18it/s]


    2026-04-09 03:09:51,945 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 03:09:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.96it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.93it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.58it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.58it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.62it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.62it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.62it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.62it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.62it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.62it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.62it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.06it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.06it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.06it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.06it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.06it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.06it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.06it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.41it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.41it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.41it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.41it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.41it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.41it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.41it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.41it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.99 GB):   2%|▏         | 1/58 [00:00<00:09,  5.82it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.96 GB):   2%|▏         | 1/58 [00:00<00:09,  5.82it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=118.96 GB):   3%|▎         | 2/58 [00:00<00:07,  7.01it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.89 GB):   3%|▎         | 2/58 [00:00<00:07,  7.01it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.89 GB):   5%|▌         | 3/58 [00:00<00:07,  7.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.88 GB):   5%|▌         | 3/58 [00:00<00:07,  7.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.88 GB):   5%|▌         | 3/58 [00:00<00:07,  7.63it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.88 GB):   9%|▊         | 5/58 [00:00<00:04, 10.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.87 GB):   9%|▊         | 5/58 [00:00<00:04, 10.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.88 GB):   9%|▊         | 5/58 [00:00<00:04, 10.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.88 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.87 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.87 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.49it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.87 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.86 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.49it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.86 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.86 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.86 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.86 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.85 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.85 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.85 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.85 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.84 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.84 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.83 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.38it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=118.83 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.81 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=960 avail_mem=118.83 GB):  34%|███▍      | 20/58 [00:01<00:01, 31.19it/s] Capturing num tokens (num_tokens=896 avail_mem=118.83 GB):  34%|███▍      | 20/58 [00:01<00:01, 31.19it/s]Capturing num tokens (num_tokens=832 avail_mem=118.82 GB):  34%|███▍      | 20/58 [00:01<00:01, 31.19it/s]Capturing num tokens (num_tokens=768 avail_mem=118.82 GB):  34%|███▍      | 20/58 [00:01<00:01, 31.19it/s]Capturing num tokens (num_tokens=768 avail_mem=118.82 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.39it/s]Capturing num tokens (num_tokens=704 avail_mem=118.82 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.39it/s]Capturing num tokens (num_tokens=640 avail_mem=118.81 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.39it/s]Capturing num tokens (num_tokens=576 avail_mem=118.81 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.39it/s]Capturing num tokens (num_tokens=512 avail_mem=118.80 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.39it/s]

    Capturing num tokens (num_tokens=512 avail_mem=118.80 GB):  50%|█████     | 29/58 [00:01<00:00, 36.31it/s]Capturing num tokens (num_tokens=480 avail_mem=118.82 GB):  50%|█████     | 29/58 [00:01<00:00, 36.31it/s]Capturing num tokens (num_tokens=448 avail_mem=118.81 GB):  50%|█████     | 29/58 [00:01<00:00, 36.31it/s]Capturing num tokens (num_tokens=416 avail_mem=118.81 GB):  50%|█████     | 29/58 [00:01<00:00, 36.31it/s]Capturing num tokens (num_tokens=384 avail_mem=118.76 GB):  50%|█████     | 29/58 [00:01<00:00, 36.31it/s]Capturing num tokens (num_tokens=384 avail_mem=118.76 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=352 avail_mem=118.75 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=320 avail_mem=118.75 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=288 avail_mem=118.73 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=256 avail_mem=118.72 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.01it/s]

    Capturing num tokens (num_tokens=256 avail_mem=118.72 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.90it/s]Capturing num tokens (num_tokens=240 avail_mem=118.72 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.90it/s]Capturing num tokens (num_tokens=224 avail_mem=118.72 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.90it/s]Capturing num tokens (num_tokens=208 avail_mem=118.71 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.90it/s]Capturing num tokens (num_tokens=192 avail_mem=118.71 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.90it/s]Capturing num tokens (num_tokens=176 avail_mem=118.71 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.90it/s]Capturing num tokens (num_tokens=176 avail_mem=118.71 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=160 avail_mem=118.71 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=144 avail_mem=118.70 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=128 avail_mem=118.70 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.51it/s]

    Capturing num tokens (num_tokens=112 avail_mem=118.70 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.51it/s]Capturing num tokens (num_tokens=112 avail_mem=118.70 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.82it/s]Capturing num tokens (num_tokens=96 avail_mem=118.67 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.82it/s] Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.82it/s]

    Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.82it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.82it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 24.04it/s]Capturing num tokens (num_tokens=32 avail_mem=118.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 24.04it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:02<00:00, 24.04it/s]

    Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:02<00:00, 24.04it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.76it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.76it/s]Capturing num tokens (num_tokens=16 avail_mem=118.63 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.76it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.76it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.76it/s] Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  98%|█████████▊| 57/58 [00:02<00:00, 27.42it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  98%|█████████▊| 57/58 [00:02<00:00, 27.42it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 25.93it/s]


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
    Generated text:  Guo Yang. I'm a 30-year-old computer scientist at Google. I'm based in China. This is my third country. My current job is to develop new AI tools and technologies that solve problems for Google. This was my first country. My current job is to improve my language skills, so that I can communicate in English and Chinese. This was my second country.
    I study computer science at Tsinghua University. I'm a computer scientist who works with complex computer systems and data to create new, high-performance computer hardware. I'm also a researcher in the field of artificial intelligence.
    I live in Beijing, China
    ===============================
    Prompt: The president of the United States is
    Generated text:  35 years older than the president of Brazil. The president of Brazil is 2 times older than the president of France. If the president of the United States is currently 40 years old, what will be the president of Brazil's age when the president of the United States retires after reaching 60 years old? Let's start by determining the current ages of the presidents. The president of the United States is currently 40 years old. According to the problem, the president of Brazil is 35 years older than the president of the United States. Therefore, the president of Brazil is:
    
    \[ 40
    ===============================
    Prompt: The capital of France is
    Generated text:  _________.  A. Lille  B. Paris  C. Bordeaux  D. Lyon  E. Nice  E. Nice
    
    The capital of France is Paris. The correct answer is E. Paris. Paris is the largest city and capital of France, located on the Seine River in the southeast of the country. The other options are not capitals or cities of France. 
    
    A. Lille is the capital of France.
    B. Paris is not a city, it is a country.
    C. Bordeaux is the capital of France, not a city.
    D. Lyon is the capital of France.
    E. Nice is
    ===============================
    Prompt: The future of AI is
    Generated text:  also moving towards a new generation of AI that is specifically designed to learn and adapt to the changing needs of society. As AI is a complex and rapidly evolving field, it is essential to have a clear understanding of the current state of AI and its potential applications. This article will provide an overview of the different types of AI, the latest trends, and the potential applications of AI in various industries.
    The main types of AI include:
    - Supervised learning: This involves training a machine learning model on labeled data, where the input data and its corresponding output are provided. The model learns to recognize patterns in the data and can make predictions based on


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Character] who is [Describe your character's personality]. I'm [Describe your character's appearance]. I'm [Describe your character's hobbies and interests]. I'm [Describe your character's strengths and weaknesses]. I'm [Describe your character's goals and aspirations]. I'm [Describe your character's personality type]. I'm [Describe your character's personality type]. I'm [Describe your character's personality type]. I'm [Describe your character's personality type]. I'm [Describe your character's personality type]. I'm [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the annual Eiffel Tower Festival. It is also the seat of the French government and the country's cultural and political capital. Paris is a major cultural and economic center, with a rich history dating back to the Roman Empire and the French Revolution. The city is home to many famous landmarks, including the Louvre Museum, the Notre-Dame Cathedral, and the Arc de Triomphe. Paris is also known for its cuisine, including its famous croissants and its traditional French wine. The city is a popular tourist destination, with millions of visitors annually. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and privacy. As AI systems become more complex and sophisticated, it will be important to ensure that they are designed and implemented in a way that respects human values and promotes fairness and justice.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such
    


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
    Generated text:  [Name], and I'm [Age]. I'm a [Type of Person] who has always [What you've always been good at or enjoy doing]. I've always been passionate about [Your Hobby/Interest], which has led me to pursue [What you've chosen to do in your free time]. I enjoy [What you do for fun], and I have a [Favorite Accomplishment/Challenge]. I'm excited to meet you! I look forward to meeting you and chatting with you! - [Name] [Note: Replace [Name], [Age], [Type of Person], [What you've always been good at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and the fifth-largest city in the world by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and Montmartre. The city is also famous for its rich history and cultural heritage, including the Louvre Museum and the annual Eiffel Tower Festival. The city is a hub for business and politics, and is home to many of the country's top universities and cultural institutions. As the heart of French culture and society, Paris plays a crucial role in shaping the nation's identity and values. According to the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly promising and has the potential to revolutionize every aspect of our lives. Here are some potential trends that we can expect to see in the near future:
    
    1. Improved AI Ethics: With the increasing number of ethical dilemmas and controversies surrounding AI, it's becoming increasingly important for AI developers to prioritize ethical considerations in their designs. This could lead to the development of new AI technologies that are more transparent and accountable, and could even result in the development of new ethical guidelines for AI applications.
    
    2. More Interconnected AI: With the rapid development of interconnected networks, AI is likely to become even more integrated with other technologies, such as sensors


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

    ].

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Occup

    ation

    ]

     (

    e

    .g

    .

     lawyer

    ,

     doctor

    ,

     teacher

    ).

     My

     [

    favorite

     color

    ]

     is

     [

    Favorite

     Color

    ],

     and

     I

     have

     a

     [

    number

     of

     hobbies

    ]

     in

     common

     with

     the

     people

     around

     me

    .

     
    


    As

     an

     AI

     language

     model

    ,

     I

    'm

     programmed

     to

     be

     neutral

     and

     unbiased

    ,

     so

     I

     don

    't

     have

     any

     personal

     opinions

     or

     biases

    .

     But

     I

    'd

     love

     to

     hear

     what

     you

     think

     of

     me

    .

     What

     do

     you

     think

     of

     myself

     as

     a

     human

     being

    ?


    [

    Name

    ].

     I

     am

     a

     language

     model

     designed

     to

     assist

     with

     a

     wide

     variety

     of

     tasks

     and

     inquiries

    .

     I

     am

     programmed

     to

     be

     impartial

    
    
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

    .

     It

     is

     known

     for

     its

     museums

    ,

     parks

    ,

     and

     historic

     landmarks

    ,

     and

     is

     a

     major

     center

     for

     the

     arts

    ,

     science

    ,

     and

     technology

     industries

    .

     The

     city

     also

     hosts

     the

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

     the

     E

    iff

    el

     Tower

    ,

     and

     numerous

     museums

     and

     attractions

     throughout

     the

     city

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     and

     is

     home

     to

     many

     important

     historical

     sites

     and

     events

     such

     as

     the

     World

     Cup

    .

     
    


    Some

     of

     the

     most

     famous

     attractions

     and

     landmarks

     include

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

     the

     E

    iff

    el

     Tower

    ,

     the

     Ch

    amps

    -

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

     and

     it

     is

     expected

     to

     evolve

     rapidly

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     availability

     of

     AI

     technology

    :

     As

     the

     cost

     of

     AI

     technology

     decreases

    ,

     more

     organizations

     will

     be

     able

     to

     afford

     AI

     solutions

     for

     their

     businesses

    .

     This

     will

     lead

     to

     a

     greater

     number

     of

     AI

     applications

     and

     tools

     that

     can

     be

     used

     to

     improve

     efficiency

     and

     reduce

     costs

    .
    


    2

    .

     AI

    -driven

     automation

    :

     AI

     will

     continue

     to

     automate

     a

     significant

     portion

     of

     business

     processes

    ,

     from

     manufacturing

     to

     healthcare

    ,

     transportation

    ,

     and

     transportation

    ,

     among

     others

    .

     This

     will

     enable

     businesses

     to

     save

     time

     and

     reduce

     costs

    ,

     making

     them

     more

     competitive

     in

     the

     market

    .
    


    3

    .

     AI

    -driven

     personalized

     experiences

    



```python
llm.shutdown()
```
