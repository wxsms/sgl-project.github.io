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


    2026-04-08 04:23:50.852 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 04:23:50] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 04:23:50.853 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 04:23:50] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 04:23:50.853 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 04:23:50] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 04:23:50.853 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 04:23:50] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 04:23:50.853 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 04:23:50] Persistent cache disabled, using in-memory JIT cache


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.97it/s]


    2026-04-08 04:23:53,683 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 04:23:53] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.65it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.65it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.65it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.96it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.96it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.54it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.43it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.54it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.54it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.54it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.54it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.54it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.54it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.54it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 46.54it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 17.98it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.98it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.98it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.92it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.92it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.27it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.27it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.24it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.24it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 19.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  34%|███▍      | 20/58 [00:00<00:01, 19.21it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 19.21it/s] Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 19.21it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.21it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.04it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.04it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.04it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.04it/s]Capturing num tokens (num_tokens=576 avail_mem=120.27 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.04it/s]

    Capturing num tokens (num_tokens=512 avail_mem=120.26 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.04it/s]Capturing num tokens (num_tokens=512 avail_mem=120.26 GB):  50%|█████     | 29/58 [00:01<00:01, 28.33it/s]Capturing num tokens (num_tokens=480 avail_mem=120.27 GB):  50%|█████     | 29/58 [00:01<00:01, 28.33it/s]Capturing num tokens (num_tokens=448 avail_mem=119.05 GB):  50%|█████     | 29/58 [00:01<00:01, 28.33it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:01<00:01, 28.33it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:01<00:01, 28.33it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  50%|█████     | 29/58 [00:01<00:01, 28.33it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.35it/s]

    Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.55it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.55it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.55it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.55it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.55it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.55it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.84it/s]

    Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.84it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.79it/s]

    Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.79it/s] Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 32.30it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 29.73it/s]


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
    Generated text:  David and I am a 22-year-old man. I work as a freelancer on a platform called Upwork and I earn a monthly income of $2,500. I've been freelancing since I was 20 and have learned a lot about marketing, networking, and the creative process. My goal is to use my platform to help others. 
    
    I’ve been in the industry since I was 18 and I have over 500 online hours. I have a wide range of skills, and I can handle the different types of jobs and clients I work with. I have a passion for storytelling and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. This is true. However, this president is not the president of the United States. What is true? (A) The president of the United States is a woman. (B) The president of the United States is a man. (C) The president of the United States is a woman. (D) The president of the United States is a man. To determine the correct answer, we need to carefully analyze the given statement: "The president of the United States is a man. This is true." The president of the United States is referred to as the "president," and the statement clearly indicates that this
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    
    A: London  
    B: Paris  
    C: Moscow  
    D: St. Petersburg
    To determine the capital of France, we need to consider the following information:
    
    1. France is a country.
    2. The capital of a country is its largest city.
    3. The cities of France are located in the north of the country.
    
    Given these points, the capital of France is Paris. Therefore, the correct answer is:
    
    \boxed{B}
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain. In my view, it's not just about the advancements in AI, but also about how people interact with and use AI. As an AI researcher, I have to constantly update my knowledge base to stay ahead of the curve. In the past, AI applications were more limited, and people tended to rely on humans to make decisions. However, as we have seen in recent years, AI has become more advanced and capable, and people can now make decisions based on AI algorithms.
    
    One of the biggest challenges in the AI field is the issue of ethics. There are ethical considerations that need to be taken into account when developing AI systems.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I'm always looking for ways to [what I enjoy doing]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major center for business, finance, and tourism in Europe. Paris is a popular tourist destination and a major economic hub. The city is home to many international organizations and institutions, including the European Parliament and the United Nations. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine. Additionally, AI is likely to play an increasingly important role in areas such as healthcare, finance, and energy, as it can help to automate and optimize processes and reduce errors. However, there are also potential risks and challenges associated with AI, such as job displacement and ethical concerns, which will need to be carefully managed and addressed. Overall, the future of AI is likely to be a rapidly evolving and complex
    


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
    Generated text:  [Name], and I'm a [role] [career]. I'm an accomplished [field] [career], [特长] [accomplishment]. I'm enthusiastic about [career] and have been [introduction] for [duration].
    
    I'm passionate about [career] because [specific reason], and I'm always willing to learn and grow as a [career]. I'm known for my [career] skills, [特长] [accomplishment], and [career] knowledge. I'm always eager to share my experiences and learn from others. I'm excited to share my [career] journey with anyone who is interested
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as "The City of Light" and "The City of Light" is a synonym for the French capital. 
    
    What is the name of the city that was founded by Charles V in 1477 and is the largest city in France by area? To find the answer, I will perform a web search.
    
    The name of the city is Paris. It was founded by Charles V in 1477 and is the largest city in France by area. 
    
    Paris is the capital of France and is known as "The City of Light" and "The City of Light" is a synonym for the French capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve several key trends, including:
    
    1. Increased collaboration between humans and AI: One trend in AI is the increasing collaboration between humans and AI systems. This could involve developing AI that can learn from and adapt to human behavior, and understanding and integrating human emotions and perspectives into AI systems.
    
    2. Personalized AI: Another trend is the development of AI that is tailored to individual users, based on their preferences and behavior. This could involve using AI to provide personalized recommendations, enhance the user experience, and optimize the user's interactions with the system.
    
    3. Robust ethical considerations: AI is becoming more complex and controversial, with ethical


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

     aspiring

     [

    Job

     Title

    ]

     with

     [

    Number

    ]

     years

     of

     experience

    .

     I

     have

     a

     natural

     gift

     for

     [

    Mot

    iv

    ational

     Quote

    ],

     which

     has

     led

     me

     to

     achieve

     success

     in

     [

    Career

     Field

    ],

     [

    Company

     Name

    ].

     I

    'm

     always

     looking

     for

     ways

     to

     [

    Challenge

    ],

     and

     am

     always

     eager

     to

     learn

     something

     new

    .

     I

     love

     [

    Favorite

     Thing

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    Challenge

     or

     Learn

     Something

     New

    ].

     My

     strongest

     quality

     is

     [

    Strength

    ],

     and

     I

     strive

     to

     [

    Keep

     Me

     Mot

    ivated

    ]

     for

     my

     entire

     career

    .

     Thank

     you

     for

     having

     me

    .

     What

     specific

     traits

     or

     qualities

     do

     you

     think

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     city

     of

     light

    .

     It

     is

     the

     largest

     city

     in

     Europe

    ,

     located

     on

     the

     river

     Se

    ine

    ,

     which

     forms

     the

     heart

     of

     the

     city

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     the

     

    5

    th

     century

     BC

    .

     It

     is

     also

     known

     for

     its

     classical

     architecture

    ,

     Paris

    ian

     cuisine

    ,

     and

     its

     contributions

     to

     art

    ,

     music

    ,

     and

     literature

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

     famous

     landmarks

     and

     attractions

    .

     Its

     cultural

     and

     historical

     significance

     makes

     it

     a

     major

     city

     in

     France

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     exciting

    ,

     and

     there

     are

     many

     possible

     trends

     shaping

     its

     development

    .

     Here

     are

     some

     of

     the

     most

     promising

     ones

    :
    


    1

    .

     Increased

     AI

     transparency

     and

     accountability

    :

     As

     AI

     becomes

     more

     complex

    ,

     it

     will

     become

     more

     transparent

     and

     accountable

    .

     This

     means

     that

     AI

     systems

     will

     be

     more

     transparent

     about

     their

     decision

    -making

     processes

    ,

     and

     their

     outputs

     will

     be

     open

     to

     scrutiny

    .

     This

     will

     help

     to

     build

     trust

     between

     people

     and

     AI

     systems

    ,

     and

     will

     also

     help

     to

     ensure

     that

     AI

     systems

     are

     used

     in

     a

     responsible

     and

     ethical

     manner

    .
    


    2

    .

     AI

     integration

     with

     human

     beings

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     will

     see

     more

     and

     more

     AI

     systems

     being

     integrated

     into

     our

     lives

    .

     This

     could

    



```python
llm.shutdown()
```
