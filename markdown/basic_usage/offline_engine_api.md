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


    2026-04-08 06:18:13.500 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:18:13] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:18:13.500 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:18:13] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:18:13.500 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:18:13] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:18:13.500 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:18:13] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:18:13.500 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:18:13] Persistent cache disabled, using in-memory JIT cache


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.64it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.62it/s]


    2026-04-08 06:18:16,475 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 06:18:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  5.04it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  5.04it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  5.04it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:09,  5.04it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:09,  5.04it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:09,  5.04it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:09,  5.04it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:03<00:09,  5.04it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:03<00:09,  5.04it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:03<00:03, 11.48it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:03<00:03, 11.48it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 22.38it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 22.38it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 22.38it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 22.38it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 22.38it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 22.38it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 22.38it/s]

    Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 22.38it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 33.92it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 33.92it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 33.92it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 33.92it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 33.92it/s]

    Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 33.92it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 33.92it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 43.78it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 43.78it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 43.78it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 43.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.60it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.55it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.55it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.55it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.55it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 34.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 34.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 34.19it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  31%|███       | 18/58 [00:00<00:01, 34.19it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 34.19it/s] Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.77it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.77it/s]

    Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.77it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.77it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.77it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.77it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.80it/s]

    Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  71%|███████   | 41/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  71%|███████   | 41/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 35.84it/s]

    Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.78it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.13it/s]

    Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.86it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 33.33it/s]


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
    Generated text:  Remy, I'm a 17 year old female, I'm an English major and I have been trying to cut down on sugar. I think it is bad for me. So I've started cutting back on sweets, but I can't seem to get my diet straight. I have been eating lots of chocolate and I know I should be doing my best to cut back on it, but I just can't seem to get my diet straight. I know it's bad for me, but I'm having a hard time making my mind up. I'm in college and I have to take 3 classes for my final.
    I
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy with the work of the government. He has to deal with the Congress, the Executive, the Departments, and the Media. These are the seven parts of the government. The president is always busy with their tasks. He has to make important decisions. The American president is a very important person. The president's job is very important. He makes important decisions about the government. He also helps the people to decide what to do. His job is not easy. He has to deal with a lot of people and a lot of things. He has to make sure that everyone is being treated fairly. He makes sure that everyone has the
    ===============================
    Prompt: The capital of France is
    Generated text:  (  ) A. Paris B. Nice C. Nice D. Nice
    Answer:
    
    A
    
    [Multiple Choice Question] Regarding the 'Measures for the Administration of Land Appreciation Tax', which of the following statements is correct? 
    A. The tax rate for land appreciation tax is 30% 
    B. The tax base for land appreciation tax is the area occupied by the building on the land 
    C. The transfer of land use rights for real estate development projects is exempt from the land appreciation tax 
    D. The land appreciation tax is a property tax levied on the land use rights for real estate development projects 
    
    ===============================
    Prompt: The future of AI is
    Generated text:  complex, but it’s clear that it will continue to play a pivotal role in shaping our world. Here’s a look at what will change in the next few years and how it will impact society.
    Artificial intelligence (AI) is likely to be one of the most significant technological advancements of the next decade. While the technology has been around for decades, it is only in the last few years that it has begun to take on a broader scope and reach.
    The potential applications for AI are vast and varied. It can be used for everything from healthcare to finance to transportation. However, one of the biggest challenges in the development of AI is


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its cuisine, fashion, and art scene. Paris is a cultural and economic center of France and plays a significant role in the country's politics and economy. It is a popular tourist destination and a major center for international business and diplomacy. The city is home to many museums, theaters, and other cultural institutions. Paris is a vibrant and dynamic city with a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient solutions to complex problems.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more robust and transparent AI systems that are designed to minimize bias and ensure fairness.
    
    3. Greater use of
    


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
    Generated text:  [insert name]. I'm a [insert occupation or profession] and I'm [insert your current age]. I enjoy [insert interests or hobbies] and I'm always looking for new experiences to expand my knowledge and skills. I strive to be [insert your personal values or character traits], and I'm always up for [insert any activities you enjoy doing]. Thanks for having me! 
    (Note: The characters are fictional, and the identity and activities are created by the author.) 
    [Your Name]  
    [Date]  
    [Title]  
    [Location]  
    [Occupation or Profession]  
    [Current Age]  
    [Inter
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the heart of the Paris region, near the French Riviera. It is an important cultural and political center, with a rich history dating back to the 6th century. The city is known for its iconic Eiffel Tower and is the largest city in the European Union. Paris is also home to the Louvre Museum, the Notre Dame Cathedral, and many other notable landmarks. The city is known for its fashion and gastronomy, with Paris Fashion Week and Dior's red carpet. Paris is also a significant center for science, with the Center for Research and Innovation in Paris (CRIO) serving as the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to expand and evolve, driven by a variety of trends and challenges. Here are some potential future trends in artificial intelligence:
    
    1. Increased integration with other technologies: The use of AI is likely to become more integrated with other technologies such as voice and motion sensors, blockchain, and IoT. This integration will enable AI to better understand and interact with users, improving overall user experience.
    
    2. Greater reliance on AI for decision-making: As AI becomes more sophisticated, it is likely to be used for more complex tasks, such as decision-making. This will require AI to be more accurate and reliable, and to be able to adapt to


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

     Alex

    .

     I

    'm

     a

     very

     nice

     guy

     who

     has

     always

     been

     kind

     and

     consider

    ate

    .

     I

     enjoy

     helping

     people

     and

     making

     the

     world

     a

     better

     place

    .

     I

    'm

     outgoing

     and

     have

     a

     great

     sense

     of

     humor

    ,

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     make

     people

     smile

    .

     I

    'm

     friendly

     and

     approach

    able

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     new

     things

    .

     I

    'm

     confident

     in

     my

     abilities

     and

     always

     strive

     to

     do

     my

     best

    .

     And

     my

     favorite

     thing

     about

     myself

     is

     that

     I

    'm

     kind

     of

     like

     a

     clown

    ,

     with

     a

     bit

     of

     a

     com

    ical

     side

     to

     me

    ,

     so

     I

     always

     have

     a

     laugh

     and

     a

     smile

     on

     my

     face

    .

     I

    'm

     excited

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

     and

     the

     City

     of

     Light

    .

     It

     is

     the

     largest

     and

     most

     populous

     city

     in

     France

     and

     home

     to

     many

     of

     its

     world

    -f

    amous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

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

     known

     for

     its

     culture

    ,

     cuisine

    ,

     and

     fashion

     industry

    .

     
    


    Paris

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Light

    "

     because

     of

     its

     colorful

     and

     lively

     streets

    ,

     illuminated

     by

     neon

     lights

     and

     fac

    ades

     painted

     with

     neon

     signs

    .

     It

     is

     also

     known

     as

     the

     "

    City

     of

     Light

    "

     due

     to

     its

     vibrant

     nightlife

    ,

     including

     the

     famous

     "

    V

    ieux

     Paris

    "

     district

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

    :
    


    1

    .

     Further

     development

     of

     machine

     learning

     and

     deep

     learning

     algorithms

    :

     As

     these

     technologies

     continue

     to

     improve

    ,

     we

     can

     expect

     to

     see

     more

     sophisticated

     models

     that

     can

     learn

     from

     complex

     and

     diverse

     data

     sets

    .
    


    2

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     The

     development

     of

     AI

     will

     likely

     prioritize

     considerations

     of

     fairness

    ,

     privacy

    ,

     and

     accountability

    .

     As

     this

     becomes

     more

     important

    ,

     we

     can

     expect

     to

     see

     more

     robust

     ethical

     guidelines

     and

     standards

    .
    


    3

    .

     Greater

     integration

     with

     other

     industries

    :

     As

     AI

     becomes

     more

     integrated

     into

     everyday

     life

    ,

     we

     can

     expect

     to

     see

     more

     businesses

     and

     governments

     invest

     in

     AI

     solutions

     that

     can

     be

     used

     to

     improve

     productivity

    ,

     reduce

     costs

    ,

     and

     increase

    



```python
llm.shutdown()
```
