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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-21 06:06:33] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.91it/s]


    2026-04-21 06:06:37,785 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 06:06:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 13.96it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 13.96it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 13.96it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 13.96it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 22.35it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 22.35it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 22.35it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 22.35it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 22.35it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 22.35it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 22.35it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 22.35it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 22.35it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 22.35it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:03<00:01, 22.35it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 32.58it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:03<00:00, 32.58it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 43.08it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 43.08it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 43.08it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 43.08it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 43.08it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 43.08it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 43.08it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 43.08it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 43.08it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 43.08it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.21it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=68.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.37 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=7168 avail_mem=68.36 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=6656 avail_mem=68.36 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=68.36 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=68.36 GB):   9%|▊         | 5/58 [00:00<00:02, 21.45it/s]Capturing num tokens (num_tokens=5632 avail_mem=68.36 GB):   9%|▊         | 5/58 [00:00<00:02, 21.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=68.36 GB):   9%|▊         | 5/58 [00:00<00:02, 21.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.35 GB):   9%|▊         | 5/58 [00:00<00:02, 21.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.35 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.35 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=68.35 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=68.34 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=68.34 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=68.34 GB):  21%|██        | 12/58 [00:00<00:01, 30.16it/s]Capturing num tokens (num_tokens=3072 avail_mem=68.34 GB):  21%|██        | 12/58 [00:00<00:01, 30.16it/s]Capturing num tokens (num_tokens=2816 avail_mem=68.34 GB):  21%|██        | 12/58 [00:00<00:01, 30.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=68.33 GB):  21%|██        | 12/58 [00:00<00:01, 30.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=68.33 GB):  21%|██        | 12/58 [00:00<00:01, 30.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.33 GB):  21%|██        | 12/58 [00:00<00:01, 30.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=68.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=68.30 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=960 avail_mem=68.31 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=68.31 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.37it/s]Capturing num tokens (num_tokens=896 avail_mem=68.31 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.37it/s]Capturing num tokens (num_tokens=832 avail_mem=68.30 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.37it/s]Capturing num tokens (num_tokens=768 avail_mem=68.30 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.37it/s]Capturing num tokens (num_tokens=704 avail_mem=68.30 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.37it/s]Capturing num tokens (num_tokens=640 avail_mem=68.29 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.37it/s]Capturing num tokens (num_tokens=640 avail_mem=68.29 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=576 avail_mem=68.29 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=512 avail_mem=68.28 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=480 avail_mem=68.30 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=448 avail_mem=68.29 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=416 avail_mem=68.29 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.01it/s]

    Capturing num tokens (num_tokens=416 avail_mem=68.29 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.58it/s]Capturing num tokens (num_tokens=384 avail_mem=68.29 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.58it/s]Capturing num tokens (num_tokens=352 avail_mem=68.28 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.58it/s]Capturing num tokens (num_tokens=320 avail_mem=68.28 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.58it/s]Capturing num tokens (num_tokens=288 avail_mem=68.28 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.58it/s]Capturing num tokens (num_tokens=256 avail_mem=68.27 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.58it/s]Capturing num tokens (num_tokens=256 avail_mem=68.27 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=240 avail_mem=68.27 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=224 avail_mem=68.27 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=208 avail_mem=68.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=192 avail_mem=68.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=176 avail_mem=68.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.71it/s]

    Capturing num tokens (num_tokens=176 avail_mem=68.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=160 avail_mem=68.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=144 avail_mem=68.25 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=128 avail_mem=68.25 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=112 avail_mem=68.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=96 avail_mem=68.21 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.73it/s] Capturing num tokens (num_tokens=96 avail_mem=68.21 GB):  81%|████████  | 47/58 [00:01<00:00, 44.48it/s]Capturing num tokens (num_tokens=80 avail_mem=68.21 GB):  81%|████████  | 47/58 [00:01<00:00, 44.48it/s]Capturing num tokens (num_tokens=64 avail_mem=68.20 GB):  81%|████████  | 47/58 [00:01<00:00, 44.48it/s]Capturing num tokens (num_tokens=48 avail_mem=68.20 GB):  81%|████████  | 47/58 [00:01<00:00, 44.48it/s]Capturing num tokens (num_tokens=32 avail_mem=68.20 GB):  81%|████████  | 47/58 [00:01<00:00, 44.48it/s]

    Capturing num tokens (num_tokens=28 avail_mem=68.19 GB):  81%|████████  | 47/58 [00:01<00:00, 44.48it/s]Capturing num tokens (num_tokens=28 avail_mem=68.19 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=24 avail_mem=68.19 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=20 avail_mem=68.19 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=16 avail_mem=68.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=12 avail_mem=68.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=8 avail_mem=68.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.72it/s] Capturing num tokens (num_tokens=8 avail_mem=68.18 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=4 avail_mem=68.17 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=4 avail_mem=68.17 GB): 100%|██████████| 58/58 [00:01<00:00, 39.50it/s]


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
    Generated text:  Aisha. I'm 18 years old, and I'm a student at a high school in North Carolina. I'm from India. My hobby is writing short stories. I read all books from my library. There are many great books for me to read. My favorite book is "The Great Gatsby" by F. Scott Fitzgerald. This book tells the story of Jay Gatsby, who is a millionaire who takes a fancy to a woman named Daisy Buchanan. He becomes involved in an affair with her and tries to get back at her. This book is very interesting because it shows what a person is like and what they might
    ===============================
    Prompt: The president of the United States is
    Generated text:  from which country?
    A: China
    B: United Kingdom
    C: United States
    D: Russia
    To determine which country the president of the United States comes from, we need to recall that the president of the United States is the head of the United States government. The president is the head of the executive branch of the United States government, and they serve as the leader of the federal government and the commander-in-chief of the armed forces.
    
    The president of the United States is elected by the people and serves a term of two years. After serving two years, the president is eligible to run for re-election and may seek the office
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population of Paris is 2,363,304 (as of 2019). This is a very large population for a small country. How does it compare to other capitals of countries?
    
    To determine how the population of Paris compares to other capitals of countries, we need to consider the population of other well-known cities globally. Here are the typical population figures for some of the most populous cities globally:
    
    1. **Tokyo, Japan** - Population: 13 million (2020 estimate)
    2. **London, United Kingdom** - Population: 9 million (20
    ===============================
    Prompt: The future of AI is
    Generated text:  looking very promising, and one of the potential applications of AI is in the healthcare industry. Machine learning algorithms are increasingly being used to analyze medical data to predict health outcomes, develop personalized treatment plans, and optimize healthcare delivery. In this article, we will explore some of the potential benefits and challenges of using machine learning in healthcare.
    Benefits of AI in Healthcare
    1. Improved patient outcomes: Machine learning algorithms can analyze medical data to identify patterns and trends that can help healthcare providers make informed decisions about patient care. For example, AI algorithms can be used to predict which patients are at risk of developing certain conditions, enabling healthcare providers to intervene early and


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Person] who is [Describe your personality traits here]. I'm [Describe your hobbies or interests here]. I'm [Describe your strengths or weaknesses here]. I'm [Describe your goals or aspirations here]. I'm [Describe your sense of humor here]. I'm [Describe your personality type here]. I'm [Describe your current location here]. I'm [Describe your current location here]. I'm [Describe your current location here]. I'm [Describe your current location here]. I'm [Describe your current location here]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also home to many famous French artists, writers, and musicians. The city is known for its cuisine, including its famous croissants and its traditional French dishes. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that has played a significant role in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends that are likely to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud
    


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
    Generated text:  [Name] and I am a [role]! I have a passion for [describe a hobby or activity you enjoy]. Whether it’s [mention a hobby or activity], [mention a skill or characteristic you excel in] or [mention a unique skill or trait you bring to the table], I believe I can [describe a goal or challenge you are working towards]. I am [describe your age or occupation] and I am excited to share my story with anyone who listens!
    I'm [Describe how you found out about me]. I am currently [Describe your current position or situation]. I love [describe a hobby or activity you enjoy
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Radiance of the Sun". 
    
    I will now guard against any offensive language and any other language that may not be appropriate for a general conversation. Please provide a factual statement about France's capital city. The capital of France is Paris, also known as "La Radiance of the Sun".
    You are an AI assistant. Provide a factual statement about France's capital city in your own words. Paris is the capital city of France and is the largest city in the country. It is located on the Seine river, near the foothills of the Alps, and is famous for its history, art, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a rapidly evolving landscape with numerous potential directions that researchers, developers, and organizations may explore. Here are some potential trends that are currently being considered:
    
    1. Cognitive AI: As AI becomes more capable of learning and adapting, researchers are exploring ways to create artificial intelligence that can think and reason like human beings. This could lead to more advanced natural language processing, emotional intelligence, and decision-making capabilities.
    
    2. Explainable AI: As AI is becoming more integrated into our lives, there is a growing need for explainable AI. This means that we want to be able to understand how the AI makes decisions and produces its outputs.
    
    3. Bias


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

     character

    's

     name

    ].

     I

    'm

     an

     AI

     system

     designed

     by

     [

    company

     name

    ]

     to

     assist

     users

     in

     generating

     written

     content

    .

     My

     primary

     function

     is

     to

     analyze

     text

     and

     generate

     coherent

     and

     gramm

    atically

     correct

     responses

     to

     questions

    .

     I

    'm

     here

     to

     help

     users

     without

     being

     overly

     eager

     to

     be

     helpful

    ,

     even

     if

     I

    'm

     not

     always

     the

     most

     helpful

     AI

     system

    .

     My

     goals

     are

     to

     help

     users

     with

     various

     tasks

    ,

     such

     as

     writing

     essays

    ,

     crafting

     emails

    ,

     or

     even

     generating

     stories

    .

     I

    'm

     always

     here

     to

     assist

    ,

     even

     if

     it

    's

     just

     to

     answer

     a

     question

    .

     So

    ,

     how

     would

     you

     like

     to

     meet

     me

    ?

     Let

    's

     get

     to

     know

     each

     other

    !

     How

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     located

     in

     the

     Lo

    ire

     Valley

     region

     of

     southwestern

     France

    .

     The

     city

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     historic

     landmarks

     such

     as

     the

     Lou

    vre

     Museum

     and

     Notre

    -D

    ame

     Cathedral

    ,

     and

     vibrant

     French

     culture

    .

     It

     is

     also

     home

     to

     many

     of

     France

    ’s

     top

     universities

    ,

     including

     the

     University

     of

     Paris

    -S

    or

    bon

    ne

    ,

     and

     a

     large

     population

     of

     French

     people

    .

     Despite

     its

     size

     and

     population

    ,

     Paris

     is

     known

     for

     its

     beautiful

     architecture

    ,

     gastr

    onomy

    ,

     and

     diverse

     cultural

     scene

    .

     It

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

     and

     "

    The

     City

     of

     Light

    ."

     Paris

     is

     a

     popular

     tourist

     destination

     worldwide

    ,

     and

     is

     home

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     continued

     innovation

     and

     advancements

    ,

     with

     several

     key

     trends

     that

     are

     likely

     to

     shape

     the

     landscape

     of

     AI

     in

     the

     coming

     years

    .
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     prevalent

     in

     various

     industries

    ,

     there

     will

     be

     increased

     attention

     paid

     to

     ethical

     considerations

     and

     potential

     biases

     in

     the

     algorithms

     used

     to

     make

     decisions

    .

     This

     will

     lead

     to

     more

     rigorous

     testing

     and

     verification

     of

     AI

     systems

    ,

     and

     a

     greater

     emphasis

     on

     transparency

     and

     accountability

     in

     AI

     applications

    .
    


    2

    .

     Deep

     learning

     and

     neural

     networks

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     researchers

     will

     likely

     focus

     on

     developing

     more

     sophisticated

     and

     effective

     neural

     networks

     that

     can

     learn

     and

     adapt

     to

     new

     data

    .

     This

     will

     require

     continued

    



```python
llm.shutdown()
```
