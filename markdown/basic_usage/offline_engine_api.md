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
    [2026-04-23 20:04:38] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.11it/s]


    2026-04-23 20:04:42,450 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 20:04:42] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.74it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.74it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.74it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.74it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.74it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.74it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.74it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.74it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.74it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.74it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.70it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.70it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.70it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.70it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.70it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.70it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.70it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.70it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.70it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.70it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.58it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.58it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.58it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.58it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.58it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.58it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.58it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.58it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.58it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.58it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.58it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 50.20it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=117.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=117.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.26it/s]Capturing num tokens (num_tokens=6144 avail_mem=117.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.26it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=117.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=117.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=117.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=117.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=117.04 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=117.04 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=117.04 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3584 avail_mem=117.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=117.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=117.03 GB):  21%|██        | 12/58 [00:00<00:01, 29.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=117.03 GB):  21%|██        | 12/58 [00:00<00:01, 29.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=117.03 GB):  21%|██        | 12/58 [00:00<00:01, 29.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=117.02 GB):  21%|██        | 12/58 [00:00<00:01, 29.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.02 GB):  21%|██        | 12/58 [00:00<00:01, 29.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.02 GB):  21%|██        | 12/58 [00:00<00:01, 29.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.02 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=117.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.60it/s]Capturing num tokens (num_tokens=1536 avail_mem=117.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.60it/s]Capturing num tokens (num_tokens=1280 avail_mem=117.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.60it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.98 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.60it/s]

    Capturing num tokens (num_tokens=960 avail_mem=117.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.60it/s] Capturing num tokens (num_tokens=960 avail_mem=117.00 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=896 avail_mem=117.00 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=832 avail_mem=116.99 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=768 avail_mem=116.99 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=704 avail_mem=116.99 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=640 avail_mem=116.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=640 avail_mem=116.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=576 avail_mem=116.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=512 avail_mem=116.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=480 avail_mem=116.99 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]

    Capturing num tokens (num_tokens=448 avail_mem=116.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=416 avail_mem=116.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=416 avail_mem=116.98 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=384 avail_mem=116.98 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=352 avail_mem=116.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=320 avail_mem=116.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=288 avail_mem=116.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=256 avail_mem=116.96 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=256 avail_mem=116.96 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.85it/s]Capturing num tokens (num_tokens=240 avail_mem=116.96 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.85it/s]

    Capturing num tokens (num_tokens=224 avail_mem=116.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.85it/s]Capturing num tokens (num_tokens=208 avail_mem=116.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.85it/s]Capturing num tokens (num_tokens=192 avail_mem=116.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.85it/s]Capturing num tokens (num_tokens=176 avail_mem=116.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.85it/s]Capturing num tokens (num_tokens=176 avail_mem=116.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.05it/s]Capturing num tokens (num_tokens=160 avail_mem=116.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.05it/s]Capturing num tokens (num_tokens=144 avail_mem=116.34 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.05it/s]Capturing num tokens (num_tokens=128 avail_mem=116.34 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.05it/s]

    Capturing num tokens (num_tokens=112 avail_mem=116.34 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.05it/s]Capturing num tokens (num_tokens=96 avail_mem=116.33 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.05it/s] Capturing num tokens (num_tokens=96 avail_mem=116.33 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=80 avail_mem=116.33 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=64 avail_mem=116.32 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=48 avail_mem=116.32 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=32 avail_mem=116.32 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=28 avail_mem=116.31 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=28 avail_mem=116.31 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.77it/s]Capturing num tokens (num_tokens=24 avail_mem=116.31 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.77it/s]Capturing num tokens (num_tokens=20 avail_mem=116.30 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.77it/s]

    Capturing num tokens (num_tokens=16 avail_mem=116.30 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.77it/s]Capturing num tokens (num_tokens=12 avail_mem=116.30 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.77it/s]Capturing num tokens (num_tokens=8 avail_mem=116.30 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.77it/s] Capturing num tokens (num_tokens=8 avail_mem=116.30 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=4 avail_mem=116.29 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=4 avail_mem=116.29 GB): 100%|██████████| 58/58 [00:01<00:00, 36.68it/s]


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
    Generated text:  Lillian and I'm a patient of the University of Southern California.
    My current interest is in geriatrics, as I am interested in learning about ways of preventing the progression of age-related diseases, and how to slow that progression. I am also interested in human aging research and ethics. I have completed my PhD in Gerontology from USC in 2018.
    My research area is aging and aging research ethics. I'm a member of the USC Aging Studies Research Group. The group works to develop research practices, policies, and tools that support the field of aging research ethics and advance the field of gerontology research. We are conducting
    ===============================
    Prompt: The president of the United States is
    Generated text:  a public office. This is a constitutional position, and the people elect the president. The president is the head of government, and is the commander-in-chief of the armed forces. President can only be removed by impeachment or by Congress.
    
    I don't want to let anyone change the constitution, because I need to write a paper on it. I'm not going to use any articles or sources that are outside the constitution. Please write a paper that includes:
    
    1. Overview of the Constitution
    2. The President of the United States
    3. The current President
    4. The current challenges faced by the President
    5. The opposition to
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    
    B) Lille
    
    C) Strasbourg
    
    D) Rennes
    
    To determine the capital of France, let's review the options and systematically eliminate the incorrect choices:
    
    A) Paris: This is the capital of France. The correct capital city of France is Paris.
    
    B) Lille: This is a city in France, but it is not the capital.
    
    C) Strasbourg: This is a city in France, but it is not the capital.
    
    D) Rennes: This is a city in France, but it is not the capital.
    
    The correct answer is A) Paris.
    
    Thus, the capital
    ===============================
    Prompt: The future of AI is
    Generated text:  inevitable and ubiquitous, but how can we keep it under control? The ultimate goal is to empower our society and make it a safe and stable place for everyone. Here are the top 5 ways to ensure the future of AI is peaceful and secure:
    
    1. Foster a culture of innovation and collaboration: Encouraging the exchange of ideas, collaboration, and innovation is essential to ensure that the development of AI is both efficient and safe. Encouraging the development of open-source AI platforms and tools can also help to ensure that the technology is freely accessible to everyone.
    
    2. Protect privacy: As AI continues to become more advanced, it is


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


    Generated text:  [Name] and I am a [occupation] who has been [number of years] in the industry. I am passionate about [reason for passion], and I am always looking for ways to [action or achievement]. I am a [type of person] who is [character trait or quality]. I am [character description]. I am [personality type]. I am [age] years old. I am [occupation] and I am [number of years] in the industry. I am passionate about [reason for passion], and I am always looking for ways to [action or achievement]. I am a [type of person]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its cuisine, including its famous croissants and its traditional French cuisine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city that has played a significant role in the development of French culture and is considered one of the most beautiful
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation and efficiency: AI is expected to continue to automate tasks and reduce the need for human intervention, leading to increased efficiency and productivity. This could result in new job roles being created or existing ones being replaced.
    
    2. Enhanced human-machine collaboration: AI is likely to become more integrated with human-machine interactions, allowing for more complex and nuanced decision-making. This could lead to a more human-like experience for users.
    
    3. AI-driven healthcare advancements: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI technology continues to advance, we can expect
    


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
    Generated text:  [Your Name], and I'm a [Current Occupation or Profession] at [Your Company] who has been working in this field for [Number of Years] years. I'm passionate about [Reason for Passion] and always strive to do my best work. I'm a [Any relevant skills or hobbies] who have a strong work ethic and pride in my work. I'm confident in my ability to solve complex problems and deliver results. My goal is to [Your Goal for the Month or Year] and I'm committed to always doing my best and contributing to the team. Thank you for taking the time to learn more about me.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France, located on the western bank of the Seine river, and is the seat of the French government, the national capital. It is also a cultural, intellectual, and political center with a diverse population of over 1. 4 million people. The city has a rich history and a rich cultural heritage, and is renowned for its architecture, art, cuisine, and music. It is also a popular tourist destination, with over 8 million visitors per year. The city is known for its annual Eiffel Tower, as well as its historical landmarks and museums. Paris is the third-largest
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by increasing sophistication, automation, and integration with human expertise. Here are some possible trends that we can expect to see in the future of AI:
    
    1. Improved Emotional Intelligence: With the help of machine learning algorithms, AI can learn to understand and empathize with human emotions, and provide more personalized and empathetic responses to users.
    
    2. Enhanced Natural Language Processing: Natural language processing (NLP) has become more sophisticated, allowing AI to understand and generate human-like text with greater accuracy.
    
    3. Autonomous Vehicles: Self-driving cars are already in use and will continue to become more common as AI technology improves. Autonomous vehicles


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

     an

     [

    occupation

    ]

     [

    Type

     of

     Occupation

    ],

     and

     I

    've

     always

     been

     passionate

     about

     [

    X

    ].

     
    


    I

    've

     always

     been

     fascinated

     by

     [

    X

    ],

     and

     I

    've

     always

     wanted

     to

     learn

     more

     about

     it

    .

     I

    've

     been

     reading

     a

     lot

     about

     it

    ,

     and

     I

    've

     been

     inspired

     by

     the

     people

     who

     have

     discovered

     it

    .

     I

     know

     that

     this

     is

     not

     easy

    ,

     but

     I

     know

     that

     I

     have

     the

     drive

     and

     the

     ambition

     to

     succeed

     in

     my

     pursuit

    .

     
    


    I

     hope

     that

     you

    're

     ready

     to

     learn

     more

     about

     [

    X

    ],

     and

     I

     look

     forward

     to

     hearing

     from

     you

    .

     
    


    [

    Name

    ]

     [

    Date

    ]

     [

    Location

    ]

     [

    Occup

    ation

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     The

     city

     is

     home

     to

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

     the

     Notre

     Dame

     Cathedral

    ,

     and

     many

     other

     iconic

     landmarks

    .

     It

     is

     also

     the

     birth

    place

     of

     French

     literature

     and

     a

     popular

     tourist

     destination

    .

     
    


    Paris

    ,

     often

     referred

     to

     as

     the

     "

    City

     of

     Light

    ,"

     is

     a

     vibrant

     city

     with

     a

     rich

     history

     dating

     back

     thousands

     of

     years

    .

     The

     city

     has

     undergone

     many

     transformations

    ,

     including

     the

     construction

     of

     the

     first

     bridge

    ,

     the

     rise

     of

     the

     French

     Revolution

    ,

     and

     the

     French

     Revolution

     itself

    .

     Today

    ,

     Paris

     is

     a

     bustling

     met

    ropolis

     with

     an

     international

     community

    ,

     offering

     a

     diverse

     range

     of

     cultural

     experiences

     and

     a

     rich

     tape

    stry

     of

     traditions

    .

     It

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     different

     trends

    ,

     as

     new

     technologies

     and

     developments

     continue

     to

     emerge

    .

     Here

     are

     some

     of

     the

     most

     common

     trends

     that

     are

     likely

     to

     shape

     the

     AI

     landscape

     in

     the

     years

     to

     come

    :
    


    1

    .

     Increased

     focus

     on

     ethics

     and

     transparency

    :

     As

     more

     people

     become

     aware

     of

     the

     negative

     impacts

     of

     AI

    ,

     there

     is

     a

     growing

     need

     for

     it

     to

     be

     more

     transparent

     and

     ethical

    .

     This

     could

     lead

     to

     increased

     focus

     on

     ethical

     considerations

     and

     the

     development

     of

     guidelines

     for

     the

     responsible

     use

     of

     AI

    .
    


    2

    .

     Integration

     of

     AI

     into

     healthcare

     and

     medicine

    :

     As

     AI

     continues

     to

     be

     used

     in

     healthcare

     and

     medicine

    ,

     we

     may

     see

     a

     shift

     towards

     using

     it

     to

    



```python
llm.shutdown()
```
