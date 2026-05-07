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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.41it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.40it/s]


    2026-05-07 09:35:54,427 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 09:35:54] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.43it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.43it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.93it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.95it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.94it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.94it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.94it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.94it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.94it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.94it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.94it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 15.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 15.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:03, 15.88it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.02 GB):   7%|▋         | 4/58 [00:00<00:03, 16.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.01 GB):   7%|▋         | 4/58 [00:00<00:03, 16.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.00 GB):   7%|▋         | 4/58 [00:00<00:03, 16.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.00 GB):  10%|█         | 6/58 [00:00<00:03, 14.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.99 GB):  10%|█         | 6/58 [00:00<00:03, 14.09it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=70.98 GB):  10%|█         | 6/58 [00:00<00:03, 14.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.98 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.97 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.48 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.32 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.84it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=70.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.32 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.31 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.31 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.30 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.30 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.97it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=70.30 GB):  31%|███       | 18/58 [00:00<00:01, 24.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.30 GB):  31%|███       | 18/58 [00:00<00:01, 24.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.30 GB):  31%|███       | 18/58 [00:00<00:01, 24.30it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.28 GB):  31%|███       | 18/58 [00:00<00:01, 24.30it/s]Capturing num tokens (num_tokens=960 avail_mem=70.29 GB):  31%|███       | 18/58 [00:00<00:01, 24.30it/s] Capturing num tokens (num_tokens=960 avail_mem=70.29 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.88it/s]Capturing num tokens (num_tokens=896 avail_mem=70.29 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.88it/s]Capturing num tokens (num_tokens=832 avail_mem=70.28 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.88it/s]Capturing num tokens (num_tokens=768 avail_mem=70.28 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.88it/s]Capturing num tokens (num_tokens=704 avail_mem=70.28 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.88it/s]

    Capturing num tokens (num_tokens=640 avail_mem=70.27 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.88it/s]Capturing num tokens (num_tokens=640 avail_mem=70.27 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=576 avail_mem=70.27 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=512 avail_mem=70.26 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=480 avail_mem=70.27 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=448 avail_mem=70.27 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=416 avail_mem=70.27 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=416 avail_mem=70.27 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.56it/s]Capturing num tokens (num_tokens=384 avail_mem=70.27 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.56it/s]Capturing num tokens (num_tokens=352 avail_mem=70.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.56it/s]Capturing num tokens (num_tokens=320 avail_mem=70.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.56it/s]Capturing num tokens (num_tokens=288 avail_mem=70.25 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.56it/s]

    Capturing num tokens (num_tokens=256 avail_mem=70.25 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.56it/s]Capturing num tokens (num_tokens=256 avail_mem=70.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=240 avail_mem=70.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=224 avail_mem=70.24 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=208 avail_mem=70.24 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=192 avail_mem=70.24 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=176 avail_mem=70.24 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=176 avail_mem=70.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.14it/s]Capturing num tokens (num_tokens=160 avail_mem=70.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.14it/s]Capturing num tokens (num_tokens=144 avail_mem=70.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.14it/s]Capturing num tokens (num_tokens=128 avail_mem=70.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.14it/s]Capturing num tokens (num_tokens=112 avail_mem=70.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.14it/s]

    Capturing num tokens (num_tokens=96 avail_mem=70.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.14it/s] Capturing num tokens (num_tokens=96 avail_mem=70.22 GB):  81%|████████  | 47/58 [00:01<00:00, 43.38it/s]Capturing num tokens (num_tokens=80 avail_mem=70.22 GB):  81%|████████  | 47/58 [00:01<00:00, 43.38it/s]Capturing num tokens (num_tokens=64 avail_mem=70.22 GB):  81%|████████  | 47/58 [00:01<00:00, 43.38it/s]Capturing num tokens (num_tokens=48 avail_mem=70.21 GB):  81%|████████  | 47/58 [00:01<00:00, 43.38it/s]Capturing num tokens (num_tokens=32 avail_mem=70.21 GB):  81%|████████  | 47/58 [00:01<00:00, 43.38it/s]Capturing num tokens (num_tokens=28 avail_mem=70.20 GB):  81%|████████  | 47/58 [00:01<00:00, 43.38it/s]Capturing num tokens (num_tokens=28 avail_mem=70.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.15it/s]Capturing num tokens (num_tokens=24 avail_mem=70.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.15it/s]Capturing num tokens (num_tokens=20 avail_mem=70.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.15it/s]Capturing num tokens (num_tokens=16 avail_mem=70.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.15it/s]Capturing num tokens (num_tokens=12 avail_mem=70.19 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.15it/s]

    Capturing num tokens (num_tokens=8 avail_mem=70.19 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.15it/s] Capturing num tokens (num_tokens=8 avail_mem=70.19 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=4 avail_mem=70.19 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=4 avail_mem=70.19 GB): 100%|██████████| 58/58 [00:01<00:00, 32.43it/s]


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
    Generated text:  Ahmad and I am the creator of the website www.ajdi.re. I am a programmer, an author, a gamer, and a student of architecture at the University of the Aegean (http://www.uoa.gr/), in Zakynthos. My design is based on the principles of modern art and I am interested in the artistic creation and the information design. I am a member of the "The Association of Modern Art Students" (http://www.ajdi.org/).
    I am the author of "The Joy of Architecture" (http://www.ajdi.re/the-joy-of-architecture.html). It
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking an exit interview to discuss the upcoming budget. The exit interview will take place on Thursday, November 11th, and is scheduled to last one hour. If the president arrives at the White House at 9:00 AM, what time will he arrive at the exit interview? If the exit interview is 3 hours away, what time will he arrive at the exit interview? To determine the time the president will arrive at the exit interview, we need to consider the following:
    
    1. The president arrives at the White House at 9:00 AM.
    2. The exit interview is scheduled to last one hour.
    
    
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lyon
    C. Lille
    D. Brussels
    
    The correct answer is A, Paris. The capital city of France is Paris. The other options, Lyon, Lille, and Brussels, are not capital cities of France. The third capital city, Brussels, is not a regular capital and is an important center of the European Union. The second capital city, Lille, is a city in the Loire Valley and is known for its historical and cultural significance. The first capital city, Paris, is the capital of France. The cities of Lyon, Lille, and Brussels are all not
    ===============================
    Prompt: The future of AI is
    Generated text:  a complex one. While it’s clear that AI is transforming virtually every sector of the economy, it is not clear how the impact will be felt in the future. On the one hand, it will be a profound change in how people work. On the other hand, it will be a profound change in how we live our lives. The question that we must all ask ourselves is: Are we ready for this change? In order to address this, we need to consider several factors before and during the development of AI. There are several issues that need to be considered, such as the impact of bias and privacy concerns, the need for education


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the annual Eiffel Tower Festival. It is also the seat of the French government and the country's cultural and political center. Paris is a bustling metropolis with a rich history dating back to the Roman Empire and the French Revolution. The city is home to many famous landmarks and museums, including the Louvre and the Notre-Dame Cathedral. Paris is also known for its food, fashion, and music scenes, making it a popular tourist destination. The city is home to many international organizations and organizations, including the United Nations and the World Trade Organization. Paris is a vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations and guidelines for its development and use. This could lead to more rigorous testing and regulation of AI
    


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
    Generated text:  [name] and I am [age]. I'm [occupation] and I'm a [type of character]. What brings you to this world? How has your journey so far been? And how do you plan to make the most of your future? I’m looking forward to meeting you! 🚀✨✨
    
    This introduction aims to be concise and to the point, aiming to capture the character's essence and the reader's interest in their story. It should be professional yet welcoming, reflecting your confidence and your desire to engage with potential readers. The tone is casual and approachable, allowing for a more relaxed introduction, without the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its rich history, diverse culture, and stunning architecture. Paris is a bustling metropolis with a rich cultural heritage that is reflected in its iconic landmarks such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. The city is also a vibrant hub of business, education, and entertainment, attracting visitors from around the world. Paris has a unique blend of old and new, and it is a city that continues to thrive in the modern world. Its status as the capital of France is recognized worldwide, and it plays a significant role in the country’s cultural and political life.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of different trends that are currently being researched and developed by the technology industry. Here are some possible trends:
    
    1. Advancements in machine learning: AI systems are becoming more and more capable of learning from data and adapting to new situations. This means that more complex algorithms will be needed to develop and optimize AI systems.
    
    2. Integration with human beings: The use of AI is likely to become more integrated into human society, with AI systems providing support and assistance to humans in a variety of situations. This could lead to a shift in the way we think about work, education, and healthcare.
    
    3. Personalization:


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

    ]

     and

     I

     am

     a

    /an

     [

    job

     title

    ]

     with

     over

     [

    number

    ]

     of

     [

    number

    ]

     years

     of

     experience

    .

     I

     am

     passionate

     about

     [

    job

     title

    ]

     and

     always

     aim

     to

     exceed

     expectations

    .

     [

    Name

    ]

     has

     a

     natural

     talent

     for

     problem

    -solving

     and

     a

     strong

     work

     ethic

    .

     I

     am

     always

     ready

     to

     lend

     a

     helping

     hand

     and

     help

     others

     grow

    .

     I

    'm

     excited

     to

     share

     my

     knowledge

     and

     skills

     with

     anyone

     interested

     in

     learning

    .

     


    Can

     you

     tell

     me

     about

     a

     time

     when

     you

     had

     to

     solve

     a

     complex

     problem

    ?

     I

    'm

     interested

     in

     this

     story

    ,

     but

     I

    'm

     not

     sure

     how

     to

     start

     my

     self

    -int

    roduction

    .


    Absolutely

    !

     Starting

     a

     self

    -int

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     an

     ancient

     and

     historic

     city

     located

     in

     the

     north

     of

     the

     country

    ,

     surrounded

     by

     mountains

     and

     known

     for

     its

     architecture

    ,

     cuisine

    ,

     and

     cultural

     events

    .

     The

     city

     is

     the

     third

     largest

     city

     in

     Europe

     and

     has

     a

     population

     of

     over

     

    2

    .

    7

     million

     people

    .

     Paris

     is

     also

     known

     as

     the

     City

     of

     Love

     and

     the

     City

     of

     Light

    ,

     with

     its

     many

     monuments

     and

     famous

     landmarks

    .

     It

     is

     the

     seat

     of

     government

     and

     the

     heart

     of

     the

     French

    -speaking

     world

    ,

     hosting

     numerous

     festivals

    ,

     concerts

    ,

     and

     cultural

     events

     throughout

     the

     year

    .

     Paris

     is

     also

     known

     for

     its

     café

     culture

     and

     its

     famous

     bou

    lev

    ards

     and

     p

    ia

    zz

    as

    .

     It

     is

     a

     popular

     tourist

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     rapidly

     evolving

     landscape

    ,

     with

     numerous

     potential

     areas

     of

     innovation

     and

     development

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

     Increased

     focus

     on

     ethical

     AI

    :

     As

     more

     businesses

     and

     governments

     begin

     to

     realize

     the

     importance

     of

     ethical

     AI

    ,

     there

     will

     be

     increased

     focus

     on

     creating

     AI

     that

     is

     transparent

    ,

     accountable

    ,

     and

     unbiased

    .
    


    2

    .

     Greater

     reliance

     on

     AI

     for

     decision

    -making

    :

     AI

     will

     become

     more

     integrated

     into

     business

     operations

     and

     decision

    -making

     processes

    ,

     enabling

     more

     complex

     and

     automated

     decision

    -making

    .
    


    3

    .

     AI

     will

     become

     more

     prevalent

     in

     healthcare

    :

     AI

     can

     be

     used

     to

     analyze

     medical

     data

     and

     identify

     patterns

     that

     can

     help

     diagnose

     and

     treat

     diseases

    .

     This

     could

     lead

    



```python
llm.shutdown()
```
