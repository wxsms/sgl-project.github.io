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


    2026-04-08 01:58:12.304 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 01:58:12] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 01:58:12.304 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 01:58:12] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 01:58:12.305 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 01:58:12] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 01:58:12.305 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 01:58:12] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 01:58:12.305 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 01:58:12] Persistent cache disabled, using in-memory JIT cache


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.80it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.78it/s]


    2026-04-08 01:58:15,235 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 01:58:15] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.06it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.06it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.06it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.06it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  6.06it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  6.06it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.06it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.06it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.06it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.27it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.27it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.24it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.24it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.24it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.24it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.24it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.24it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.24it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.24it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.73it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.66it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.45it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.45it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.00it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 17.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.80it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.50it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 26.22it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.32 GB):  21%|██        | 12/58 [00:00<00:01, 26.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.32 GB):  21%|██        | 12/58 [00:00<00:01, 26.22it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.10 GB):  21%|██        | 12/58 [00:00<00:01, 26.22it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=120.10 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=119.65 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=119.21 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.57it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.08 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.57it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=119.08 GB):  31%|███       | 18/58 [00:00<00:02, 17.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.98 GB):  31%|███       | 18/58 [00:00<00:02, 17.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.97 GB):  31%|███       | 18/58 [00:00<00:02, 17.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  31%|███       | 18/58 [00:00<00:02, 17.16it/s]Capturing num tokens (num_tokens=960 avail_mem=118.97 GB):  31%|███       | 18/58 [00:01<00:02, 17.16it/s] Capturing num tokens (num_tokens=960 avail_mem=118.97 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.09it/s]Capturing num tokens (num_tokens=896 avail_mem=118.97 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.09it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.09it/s]Capturing num tokens (num_tokens=768 avail_mem=118.96 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.09it/s]Capturing num tokens (num_tokens=704 avail_mem=118.96 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.09it/s]

    Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.09it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.36it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.36it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.36it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.36it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.36it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.36it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.56it/s]

    Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.06it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.06it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.06it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.06it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.06it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.06it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.90it/s]Capturing num tokens (num_tokens=160 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.90it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.90it/s]

    Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.90it/s]Capturing num tokens (num_tokens=112 avail_mem=118.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.90it/s]Capturing num tokens (num_tokens=112 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.61it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.61it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.61it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.61it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.61it/s]Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.61it/s]Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.54it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.54it/s]

    Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.54it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.54it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.54it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.54it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.93it/s] Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 30.36it/s]


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
    Generated text:  Alex and I'm a freelance engineer based out of Seattle, Washington. My expertise is in Artificial Intelligence and Machine Learning. I love to solve complex problems and help people in their pursuit of success.
    Currently, I'm working on a project to create a smart home assistant that can help with setting schedules, managing tasks, and even cooking meals. I'm hoping to see what kind of projects this assistant could help with and how it could revolutionize the way we live and work.
    I'm always looking for new ideas and fun challenges to solve, so if you have any questions or ideas for projects, don't hesitate to ask! Let's create
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. The president of the United States lives in a big house. The president's house is very big. How big? The president's house is 179 feet by 83 feet. He has 2 bedrooms, 2 bathrooms, and a kitchen. He also has a large garage and a huge swimming pool. The president also owns a library. It is the largest library in the world. It contains 13, 000, 000 books! 
    
    The president's office is very big. He has a desk and a phone. He also has a huge desk lamp. The president
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of the United States is Washington D. C.
    Is the following statement true?
    "Paris and Washington D. C. are the same place."
    Options:
    - no
    - yes
    
    no
    
    The two places have different capital cities, Paris and Washington D. C. and are located in different countries, France and the United States. Paris is the capital of France, while Washington D. C. is the capital of the United States. These cities are not the same place. The American city of Washington D. C. is in the state of D. C., which is a different state from France, the country
    ===============================
    Prompt: The future of AI is
    Generated text:  moving in the direction of a more standardized and uniform approach to algorithms that align with the new United States Cybersecurity Framework (USCF). This is a system that requires agencies to make decisions based on the requirements that are outlined in the framework. The framework includes five key areas of security – capability, risk, architecture, process, and resources. As the technology moves towards greater integration, it will also become increasingly important to implement a framework that integrates these five areas.
    Many of these areas are becoming more integrated and some of them are even being merged. For example, with the launch of the USCF, the Department of Defense (DOD)


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character here, such as "funny, adventurous, or creative"]. I enjoy [insert a short description of your character's interests or hobbies, such as "reading, playing sports, or cooking"]. I'm always looking for new experiences and challenges, and I'm always eager to learn and grow. What's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination, with millions of visitors annually. The city is known for its rich history, art, and cuisine, and is a major hub for international business and diplomacy. It is often referred to as the "City of Light" and is a UNESCO World Heritage site. Paris is a vibrant and diverse city with a rich cultural heritage, and is a major player in global affairs.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Enhanced machine learning capabilities: Machine learning algorithms are likely to become even more powerful and capable, allowing AI systems to learn from large amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective use of AI in various industries.
    
    3. Improved ethical considerations
    


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
    Generated text:  [Name], and I'm a [Age] year-old [Gender] who lives [Your City] [Country]. I have always been fascinated by [A significant area of your expertise or interest]. However, I've never been very good at [an area of your life that could be described as a weakness]. But I'm really good at [my other significant area of expertise or interest]. I've been [an activity] for [number of years] and I've [number of projects] in progress. I have [a particular hobby or interest], and I love [how I feel about that activity]. I have [how I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city famous for its fashion, food, and architecture. It was founded as a trade port on the Seine River in 869 CE. The city has a rich history and is home to many famous landmarks and attractions, such as the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées. The French language is also spoken in the city, and it is considered one of the most spoken languages in the world. Paris is a cultural hub, with many museums, theaters, and restaurants, and is the birthplace of the modern French Republic. It's the world's most visited city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by the rapid development of new technologies, improvements in data analysis and machine learning algorithms, and advances in neuromorphic computing. Here are some possible future trends in AI:
    
    1. Autonomous vehicles: Autonomous vehicles will become more common, and will drive a significant portion of human-driven transportation. This will reduce traffic congestion, accidents, and pollution.
    
    2. Smart cities: AI will enable cities to optimize energy use, infrastructure, and services, reducing costs and improving quality of life.
    
    3. Cybersecurity: AI will become more sophisticated, and will be used to protect against cyberattacks and protect the privacy of personal data.
    
    4


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

    character

    's

     occupation

    ].

     I

     have

     a

     keen

     interest

     in

     [

    mention

     something

     specific

     about

     the

     character

    ,

     such

     as

     reading

    ,

     music

    ,

     or

     art

    ].

     I

     love

     to

     travel

     and

     explore

     new

     places

    ,

     and

     I

    've

     always

     been

     fascinated

     by

     the

     world

    's

     diversity

    .

     I

    'm

     always

     on

     the

     lookout

     for

     new

     experiences

     and

     learn

     from

     every

     new

     encounter

    .

     I

    'm

     looking

     forward

     to

     meeting

     you

     and

     chatting

     with

     you

     about

     all

     things

     [

    mention

     something

     specific

     about

     the

     character

    ,

     such

     as

     books

    ,

     movies

    ,

     or

     hobbies

    ].

     I

    'm

     confident

     in

     my

     abilities

     to

     provide

     an

     engaging

     and

     informative

     experience

    .

     Thank

     you

     for

     asking

    !

     How

     can

     I

     help

     you

     today

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     rich

     history

     and

     vibrant

     culture

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     numerous

     tourist

     attractions

     and

     a

     diverse

     array

     of

     cuisine

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

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     In

     terms

     of

     language

    ,

     French

     is

     the

     official

     language

     of

     the

     country

    .

     Paris

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

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Additionally

    ,

     the

     city

     is

     known

     for

     its

     unique

     cultural

     scene

    ,

     with

     many

     theaters

    ,

     museums

    ,

     and

     festivals

    .

     Ultimately

    ,

     Paris

     is

     a

     city

     that

     is

     steep

    ed

     in

     history

    ,

     culture

    ,

     and

     vibrant

     life

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     rapidly

     evolving

    ,

     with

     many

     potential

     developments

     and

     applications

    .

     Some

     potential

     trends

     in

     AI

     include

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

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     help

     diagnose

     and

     treat

     diseases

    ,

     and

     in

     the

     development

     of

     new

     treatments

    .

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     may

     see

     even

     more

     advanced

     medical

     applications

     in

     the

     future

    .
    


    2

    .

     Personal

    ized

     AI

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     may

     see

     the

     development

     of

     AI

     that

     can

     be

     tailored

     to

     a

     person

    's

     unique

     needs

     and

     characteristics

    ,

     providing

     more

     personalized

     and

     effective

     solutions

    .
    


    3

    .

     AI

     in

     transportation

    :

     As

     autonomous

     vehicles

     become

     more

     popular

    ,

     AI

     will

     play

     an

     increasingly

    



```python
llm.shutdown()
```
