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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.17it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.17it/s]


    2026-05-10 05:42:09,240 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 05:42:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:47,  1.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:47,  1.15it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:47,  1.15it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:47,  1.15it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:47,  1.15it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.61it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.61it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.61it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.61it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.61it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.61it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.61it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.61it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.61it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.08it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.08it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.08it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.08it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.08it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.08it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.08it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 12.08it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 12.08it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:04<00:02, 12.08it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 19.54it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 19.54it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 19.54it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 19.54it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:05<00:01, 19.54it/s]Compiling num tokens (num_tokens=176):  55%|█████▌    | 32/58 [00:05<00:01, 19.54it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 29.08it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 39.01it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 39.01it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 39.01it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 39.01it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 39.01it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 39.01it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 39.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.07it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.87 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.87 GB):   3%|▎         | 2/58 [00:00<00:04, 13.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.98 GB):   3%|▎         | 2/58 [00:00<00:04, 13.26it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.98 GB):   3%|▎         | 2/58 [00:00<00:04, 13.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.98 GB):   7%|▋         | 4/58 [00:00<00:03, 13.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.97 GB):   7%|▋         | 4/58 [00:00<00:03, 13.57it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.90 GB):   7%|▋         | 4/58 [00:00<00:03, 13.57it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=71.90 GB):  10%|█         | 6/58 [00:00<00:04, 11.94it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.95 GB):  10%|█         | 6/58 [00:00<00:04, 11.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.94 GB):  10%|█         | 6/58 [00:00<00:04, 11.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.94 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.94 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.59it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.92 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.92 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.63it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.93 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.92 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.63it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.91 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.63it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.91 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.91 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.73it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=71.90 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.73it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.90 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.73it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.90 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.87 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.88 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.87 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.95it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.88 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.95it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.88 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.86 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.86it/s]

    Capturing num tokens (num_tokens=960 avail_mem=71.87 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.86it/s] Capturing num tokens (num_tokens=896 avail_mem=71.87 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=832 avail_mem=71.86 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=832 avail_mem=71.86 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.95it/s]Capturing num tokens (num_tokens=768 avail_mem=71.85 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.95it/s]Capturing num tokens (num_tokens=704 avail_mem=71.84 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.95it/s]Capturing num tokens (num_tokens=640 avail_mem=71.84 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.95it/s]Capturing num tokens (num_tokens=576 avail_mem=71.83 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.95it/s]

    Capturing num tokens (num_tokens=576 avail_mem=71.83 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.91it/s]Capturing num tokens (num_tokens=512 avail_mem=71.81 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.91it/s]Capturing num tokens (num_tokens=480 avail_mem=71.83 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.91it/s]Capturing num tokens (num_tokens=448 avail_mem=71.80 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.91it/s]Capturing num tokens (num_tokens=416 avail_mem=71.80 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.91it/s]Capturing num tokens (num_tokens=416 avail_mem=71.80 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=384 avail_mem=71.80 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=352 avail_mem=71.79 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=320 avail_mem=71.78 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.56it/s]

    Capturing num tokens (num_tokens=288 avail_mem=71.79 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=288 avail_mem=71.79 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.70it/s]Capturing num tokens (num_tokens=256 avail_mem=71.79 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.70it/s]Capturing num tokens (num_tokens=240 avail_mem=71.78 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.70it/s]Capturing num tokens (num_tokens=224 avail_mem=71.77 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.70it/s]Capturing num tokens (num_tokens=208 avail_mem=71.77 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.70it/s]Capturing num tokens (num_tokens=208 avail_mem=71.77 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.89it/s]Capturing num tokens (num_tokens=192 avail_mem=71.76 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.89it/s]Capturing num tokens (num_tokens=176 avail_mem=71.76 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.89it/s]

    Capturing num tokens (num_tokens=160 avail_mem=71.75 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.89it/s]Capturing num tokens (num_tokens=144 avail_mem=71.74 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.89it/s]Capturing num tokens (num_tokens=144 avail_mem=71.74 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.84it/s]Capturing num tokens (num_tokens=128 avail_mem=71.42 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.84it/s]Capturing num tokens (num_tokens=112 avail_mem=71.70 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.84it/s]Capturing num tokens (num_tokens=96 avail_mem=71.69 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.84it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=71.68 GB):  76%|███████▌  | 44/58 [00:02<00:00, 30.84it/s]Capturing num tokens (num_tokens=80 avail_mem=71.68 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.30it/s]Capturing num tokens (num_tokens=64 avail_mem=71.45 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.30it/s]Capturing num tokens (num_tokens=48 avail_mem=71.46 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.30it/s]Capturing num tokens (num_tokens=32 avail_mem=71.66 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.30it/s]Capturing num tokens (num_tokens=32 avail_mem=71.66 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.94it/s]Capturing num tokens (num_tokens=28 avail_mem=71.64 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.94it/s]

    Capturing num tokens (num_tokens=24 avail_mem=71.64 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.94it/s]Capturing num tokens (num_tokens=20 avail_mem=71.63 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.94it/s]Capturing num tokens (num_tokens=20 avail_mem=71.63 GB):  93%|█████████▎| 54/58 [00:02<00:00, 25.12it/s]Capturing num tokens (num_tokens=16 avail_mem=71.63 GB):  93%|█████████▎| 54/58 [00:02<00:00, 25.12it/s]Capturing num tokens (num_tokens=12 avail_mem=71.62 GB):  93%|█████████▎| 54/58 [00:02<00:00, 25.12it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  93%|█████████▎| 54/58 [00:02<00:00, 25.12it/s] Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  98%|█████████▊| 57/58 [00:02<00:00, 26.12it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB):  98%|█████████▊| 57/58 [00:02<00:00, 26.12it/s]

    Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:02<00:00, 23.79it/s]


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
    Generated text:  my first name and I am named after the man I fell in love with. I am the first name of a major city, it is also the name of a book I am in love with.
    I am from the city where I fell in love with him. I am deeply moved by the man who fell in love with him.
    My first name is Emily.
    My second name is fallen.
    The major city is my first name and the book is my second name.
    The man I fell in love with is my first name.
    I am deeply moved by him.
    My hometown is my second name. 
    My hometown is my first name.
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. This statement is ( )
    A: Sufficient but not necessary condition
    B: Necessary but not sufficient condition
    C: Sufficient and necessary condition
    D: Neither sufficient nor necessary condition
    To determine the type of logical relationship between the president of the United States and being a person, we need to analyze the definitions and implications of the terms "president" and "person."
    
    1. **Definition of a President:**
       - The president of the United States is the head of the executive branch of the federal government.
       - The president is elected by the legislative branch of the federal government, often referred to as the
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Rome
    C. London
    D. Athens
    
    To determine the capital of France, we need to look at a list of French cities and identify the capital among them. The capital of France is Paris.
    
    Here are the steps to identify the capital of France:
    1. List of French cities: Paris, London, Rome, Athens, etc.
    2. Identify the capital: Paris is the capital of France.
    
    Therefore, the capital of France is \boxed{A}.
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be a very different place. The field of artificial intelligence is progressing rapidly, and this is leading to exciting new technologies. However, there are also concerns about the ethical implications of AI systems. Some people believe that AI will lead to a loss of jobs, while others think that it could also lead to a revolution in healthcare.
    To better understand the potential impact of AI on the future of work, it's important to consider the different types of jobs that can be affected. For example, it's possible that some tasks that are currently performed by human workers will be automated by AI, which could lead to a decrease in the number of


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. What can you tell me about yourself? I'm a [insert a short description of your personality or background]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new experiences and learning opportunities. What's your favorite hobby or activity? I love [insert a short description of your favorite activity or hobby]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite book or movie? I love [insert a short description
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in Europe by population. It is located on the Seine River and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is known for its rich history, art, and culture, and is a popular tourist destination for visitors from around the world. The city is also home to many important institutions such as the French Academy of Sciences and the French National Library. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to inspire and influence the city and its people.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some potential trends that could be expected in the future:
    
    1. Increased automation: As AI becomes more advanced, it is likely to become more capable of performing tasks that were previously done by humans. This could lead to a significant increase in automation, with machines taking on many of the more routine and repetitive tasks that are currently done by humans.
    
    2. AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, it is likely to
    


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
    Generated text:  [Your Name], and I'm a [Your Profession/Role] who has always been passionate about [Your Hobby/Interest]. I'm [Your Age], [Your Education/Experience Level], and I have a knack for [Your Skill/Ability]. I'm here to [Your Goal/Objective] and I want to say "hello" to you. Can you tell me more about yourself and what motivated you to become a [Your Profession/Role] in the first place? Remember, your introduction should be friendly and direct, and include your hobbies and interests that make you unique. Let's create a connection and build a relationship that
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic city known for its historic architecture, vibrant culture, and rich history. It is also home to the Eiffel Tower, the Louvre Museum, and many other landmarks, making it a significant and recognizable city in France. Paris is a UNESCO World Heritage site and a major tourist destination, attracting millions of visitors every year. Its status as the capital has led to the development of Paris as a major economic, political, and cultural center in France. The city is also known for its cuisine, wine, and fashion, and continues to be a major cultural and economic hub for France and the world. Paris is the birth
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising and expanding rapidly, with a variety of potential trends and applications. Here are some possible future trends in AI:
    
    1. Increased Privacy and Security: With the increasing amount of data being collected, there is a growing concern about privacy and security. There is a need to develop algorithms that are more secure and less likely to be hacked, as well as better privacy-preserving techniques.
    
    2. Personalized AI: AI is being used to create personalized experiences for users, with the goal of providing better service and more relevant results for users. Personalization can be achieved by analyzing user data and behavior to create tailored recommendations and interactions.
    
    3. Autonomous


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

    insert

     your

     name

    ],

     and

     I

    'm

     an

     aspiring

     [

    insert

     your

     profession

     or

     hobby

    ].

     My

     goal

     is

     to

     create

     [

    insert

     what

     you

     are

     passionate

     about

    ,

     such

     as

     writing

    ,

     photography

    ,

     or

     gaming

    ].

     I

    'm

     excited

     to

     share

     my

     work

     and

     share

     my

     passion

     for

     my

     field

    .

     How

     can

     I

     get

     to

     know

     you

     better

    ?

     Let

     me

     know

     how

     you

     would

     like

     to

     connect

    .

     #

    self

    int

    roduction

     #

    self

    ie

     #

    fun

     #

    gl

    itter

     #

    g

    aming

    


    Hello

    !

     My

     name

     is

     [

    insert

     your

     name

    ],

     and

     I

    'm

     an

     aspiring

     [

    insert

     your

     profession

     or

     hobby

    ].

     My

     goal

     is

     to

     create

     [

    insert

     what

     you

     are

     passionate

     about

    ,

     such

     as

     writing

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     historic

     city

     with

     a

     rich

     history

     and

     stunning

     architecture

    .
    


    What

     are

     the

     key

     characteristics

     of

     a

     successful

     startup

    ,

     and

     how

     can

     businesses

     improve

     their

     chances

     of

     success

    ?

     Provide

     examples

     and

     explain

     why

     each

     characteristic

     is

     important

    .

     Additionally

    ,

     share

     

    5

     actionable

     steps

     for

     startups

     to

     improve

     their

     chances

     of

     success

    .


    Key

     characteristics

     of

     a

     successful

     startup

     include

    :
    


    1

    .

     Clear

     vision

     and

     entrepreneurial

     spirit

    


    2

    .

     Bold

     assumptions

     and

     innovation

    


    3

    .

     Strong

     leadership

     and

     strong

     team

    


    4

    .

     Strategic

     execution

     and

     successful

     market

     entry

    


    5

    .

     Continuous

     learning

     and

     adapt

    ability

    
    


    Business

    es

     can

     improve

     their

     chances

     of

     success

     by

     identifying

     and

     implementing

     the

     following

     actions

    :
    


    1

    .

     Conduct

     market

     research

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     difficult

     to

     predict

    .

     However

    ,

     here

     are

     some

     possible

     trends

     that

     may

     occur

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

    :

     AI

     will

     continue

     to

     evolve

     and

     develop

     new

     algorithms

     and

     techniques

     that

     will

     make

     it

     more

     efficient

    ,

     accurate

    ,

     and

     capable

     of

     solving

     complex

     problems

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     will

     continue

     to

     play

     an

     increasingly

     important

     role

     in

     healthcare

    ,

     with

     the

     goal

     of

     improving

     patient

     outcomes

    ,

     reducing

     costs

    ,

     and

     increasing

     access

     to

     care

    .
    


    3

    .

     Greater

     integration

     of

     AI

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     integrate

     with

     other

     technologies

    ,

     such

     as

     the

     Internet

     of

     Things

    ,

     to

     create

     more

     intelligent

     and

     connected

     systems

    .
    


    



```python
llm.shutdown()
```
