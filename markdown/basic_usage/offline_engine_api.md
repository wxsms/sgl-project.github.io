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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-18 00:51:34] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.53it/s]


    2026-04-18 00:51:39,268 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-18 00:51:39] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:44,  2.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:44,  2.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:44,  2.89s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:44,  2.89s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:44,  2.89s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:24,  2.19it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:24,  2.19it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:24,  2.19it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:24,  2.19it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:24,  2.19it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:24,  2.19it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:24,  2.19it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:24,  2.19it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:07,  6.40it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:07,  6.40it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:07,  6.40it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:07,  6.40it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:07,  6.40it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:07,  6.40it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:07,  6.40it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:07,  6.40it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:07,  6.40it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:03, 12.33it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:03, 12.33it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:03, 12.33it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 18.30it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 25.80it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 25.80it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 25.80it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 25.80it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 25.80it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 25.80it/s]

    Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 25.80it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 25.80it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 28.89it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 34.83it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 34.83it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 34.83it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 34.83it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 34.83it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 34.83it/s]

    Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 34.83it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 34.83it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 34.83it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 42.51it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 42.51it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   2%|▏         | 1/58 [00:00<00:05,  9.68it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:05,  9.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:05,  9.68it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   9%|▊         | 5/58 [00:00<00:04, 12.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:04, 12.00it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:04, 12.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.21it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.76 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.88it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  21%|██        | 12/58 [00:00<00:02, 17.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  21%|██        | 12/58 [00:00<00:02, 17.99it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  21%|██        | 12/58 [00:00<00:02, 17.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  21%|██        | 12/58 [00:00<00:02, 17.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.74 GB):  21%|██        | 12/58 [00:00<00:02, 17.99it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  21%|██        | 12/58 [00:00<00:02, 17.99it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.10it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  34%|███▍      | 20/58 [00:01<00:01, 26.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  34%|███▍      | 20/58 [00:01<00:01, 26.39it/s]Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  34%|███▍      | 20/58 [00:01<00:01, 26.39it/s] Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  34%|███▍      | 20/58 [00:01<00:01, 26.39it/s]Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  34%|███▍      | 20/58 [00:01<00:01, 26.39it/s]Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.01it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.01it/s]

    Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.01it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.01it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.01it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.12it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.12it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.12it/s]Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.12it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.12it/s]

    Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.37it/s]Capturing num tokens (num_tokens=384 avail_mem=118.70 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.37it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.37it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.37it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.37it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.29it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.29it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.29it/s]Capturing num tokens (num_tokens=224 avail_mem=118.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.29it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.29it/s]Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 31.99it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 31.99it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 31.99it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 31.99it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 31.99it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.90it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.90it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.90it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.90it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.90it/s]Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.73it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.73it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.73it/s]Capturing num tokens (num_tokens=32 avail_mem=118.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.73it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.73it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.73it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=16 avail_mem=118.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.12it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  91%|█████████▏| 53/58 [00:02<00:00, 37.12it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  91%|█████████▏| 53/58 [00:02<00:00, 37.12it/s] Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  98%|█████████▊| 57/58 [00:02<00:00, 35.49it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  98%|█████████▊| 57/58 [00:02<00:00, 35.49it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 27.39it/s]


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
    Generated text:  Jin, a mathematician, a deep mathematician. The last year I have been pursuing my Ph.D. from the University of Paris. I have been very fortunate to work with a great team of mathematicians and it is very good to meet you. I'm looking forward to meeting you in person, so please do not hesitate to ask me questions about my research interests. I am interested in algebraic and geometric problems, and I would be happy to hear from you about my projects and the problems I'm working on. Please let me know if you have any questions. Let me know if you have any questions. I have been lucky
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to hold a presidential election that will elect him in 2024 or to hold an election that will elect him in 2020. The president has a probability distribution over the years 2020, 2024, and 2028. The probability distribution is given as follows:
    
    - 2020: Probability = 0.4, Probability = 0.3, Probability = 0.3
    - 2024: Probability = 0.2, Probability = 0.6, Probability = 0.2
    -
    ===============================
    Prompt: The capital of France is
    Generated text:  located in which country?
    
    The capital of France is Paris, which is located in France. France is a country, not a capital city.
    
    Paris:
    
    1. Is the capital city of France
    2. Is in France
    3. Is a capital city of France
    4. None of the above
    5. Does not have a capital city
    
    A capital city is the city where the head of state, government, or the executive branch of a government presides over the country or region it governs. 
    
    While Paris is the capital of France, it is not a "capital city" in the traditional sense of having a seat of
    ===============================
    Prompt: The future of AI is
    Generated text:  coming, and in this digital age, it is crucial to develop an AI program that can adapt and evolve with the world around it. One way to do this is through the integration of deep learning techniques. Deep learning is a machine learning approach that allows computers to learn complex patterns and relationships in data. It has the ability to learn from large datasets and identify patterns that may not be immediately apparent to humans.
    With deep learning, organizations can create intelligent algorithms that can learn from their data and make predictions or decisions based on that data. This ability to learn and adapt can be particularly useful in industries such as healthcare, finance, and logistics.
    To


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest] and I'm always looking for ways to [action or goal]. I'm a [reason for interest] and I'm always looking for ways to [action or goal]. I'm a [reason for interest] and I'm always looking for ways to [action or goal]. I'm a [reason for interest] and I'm always looking for ways to [action or goal]. I'm a [reason for interest
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major transportation hub, with many major highways and rail lines connecting it to other parts of France and the world. The city is known for its fashion industry, art scene, and food culture. It is a popular tourist destination and a major economic center in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare to transportation. This will lead to increased efficiency, cost savings, and job displacement, but it will also create new opportunities for innovation and creativity.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its impact on society. This includes issues such as bias in AI
    


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
    Generated text:  [name] and I'm a [role], [job title] at [company name]. I've been working for [company name] since [year] and I'm passionate about [reason for your job]. What's your name, and what can you tell me about yourself? I'm excited to meet you! 👋🏼
    I'm an [type of work] at [company name], a [industry or category] [company name]. I have [number of years] years of experience in [industry/field]. 🌐✨
    In my previous roles, I have worked on [tasks or projects], which I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is known for its stunning architecture, vibrant culture, and iconic landmarks such as the Eiffel Tower and the Louvre Museum. It is also home to the French Parliament and many of France's important museums and attractions. Paris is a cultural and political center that continues to attract visitors from all over the world. Its significance as a major city in France and the world is reflected in its many artistic, historical, and cultural events. Paris is often referred to as the "City of Love" due to its romantic and passionate atmosphere. The city is also famous for its fashion industry, and the iconic perfume Chanel. Paris is the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by several key trends:
    
    1. Deep learning and machine learning: These are the primary techniques used to develop AI systems. Deep learning is a type of machine learning that involves using neural networks with multiple layers to learn from data. Machine learning is a subset of deep learning that uses statistical models to make predictions or decisions.
    
    2. Natural language processing: This is a key area of AI research that aims to make machines understand and process human language. The goal is to create systems that can understand natural language and respond appropriately to human language.
    
    3. Autonomous vehicles: As the world becomes more aware of the environmental and safety risks associated


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

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ],

     I

     have

     [

    Occup

    ation

    ]

     and

     [

    Status

    ],

     and

     I

     enjoy

     [

    Favorite

     Activity

    ].
    


    I

     also

     enjoy

     [

    Att

    itude

     Towards

     Challenges

    ],

     but

     I

     don

    't

     care

     about

     [

    Challenge

     Name

    ].

     I

    'm

     passionate

     about

     [

    Interest

    ],

     and

     I

    'm

     always

     [

    State

     of

     Mind

    ].

     I

    'm

     [

    Positive

     or

     Negative

     Att

    itude

    ]

     and

     I

    'm

     [

    Future

     Goals

    ].

     I

    'm

     a

     [

    Type

     of

     Person

    ].

     I

    'm

     an

     [

    Experience

     of

     the

     Year

    ].

     I

    'm

     [

    What

    's

     Your

     Special

     Character

    ].

     I

    'm

     [

    Background

     Information

    ].

     I

    'm

     [

    Pro

    tag

    on

    ist

    ].

     I

    'm

     a

     [

    Character

     Name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     historic

     and

     modern

     city

     renowned

     for

     its

     rich

     cultural

     history

    ,

     iconic

     landmarks

    ,

     and

     vibrant

     city

     life

    .

     It

     is

     home

     to

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

     other

     renowned

     attractions

    .

     Its

     old

     quarter

    ,

     the

     Ch

    amps

    -

    É

    lys

    ées

    ,

     is

     also

     a

     UNESCO

     World

     Heritage

     site

    ,

     and

     it

     has

     a

     rich

     history

     dating

     back

     to

     Roman

     times

    .

     With

     its

     blend

     of

     ancient

     traditions

     and

     modern

     conven

    iences

    ,

     Paris

     is

     a

     city

     of

     ancient

     charm

     and

     modern

     glamour

    .

     The

     city

     is

     considered

     one

     of

     the

     top

     destinations

     for

     tourists

     in

     Europe

     and

     is

     known

     for

     its

     diverse

     culinary

     scene

    .

     Over

     

    7

     million

     people

     live

     in

     Paris

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     difficult

     to

     predict

    ,

     but

     there

     are

     several

     potential

     trends

     that

     could

     shape

     its

     development

     and

     impact

     in

     the

     coming

     years

    .

     Some

     of

     the

     most

     significant

     trends

     include

    :
    


    1

    .

     Increased

     AI

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     medical

     devices

    ,

     smart

     home

     systems

    ,

     and

     self

    -driving

     vehicles

    ,

     the

     potential

     for

     AI

     to

     improve

     quality

     of

     life

     and

     reduce

     human

     error

     could

     become

     even

     more

     widespread

    .
    


    2

    .

     More

     autonomous

     vehicles

    :

     With

     the

     development

     of

     AI

    -powered

     self

    -driving

     technology

    ,

     the

     future

     could

     see

     autonomous

     vehicles

     becoming

     more

     common

     in

     daily

     life

    ,

     making

     transportation

     more

     efficient

     and

     safer

    .
    


    3

    .

     AI

     for

     healthcare

    :

     AI

     is

    



```python
llm.shutdown()
```
