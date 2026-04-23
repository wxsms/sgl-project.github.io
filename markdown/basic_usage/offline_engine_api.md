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
    [2026-04-23 22:30:13] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.88it/s]


    2026-04-23 22:30:26,313 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 22:30:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.90it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.90it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.90it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.90it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.90it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.90it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.90it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.90it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.90it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.90it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:08,  5.90it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 14.19it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 14.19it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 14.19it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 14.19it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 14.19it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:02<00:02, 14.19it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:02<00:02, 14.19it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:02<00:02, 14.19it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:02<00:02, 14.19it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 14.19it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:03<00:02, 14.19it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 23.56it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 23.56it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 23.56it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 23.56it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 23.56it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 23.56it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 23.56it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 23.56it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 23.56it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 23.56it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:03<00:01, 23.56it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 33.82it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 33.82it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 33.82it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 33.82it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 33.82it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 33.82it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 33.82it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 33.82it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:03<00:00, 33.82it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:03<00:00, 33.82it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:03<00:00, 33.82it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 44.33it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 44.33it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 44.33it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 44.33it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 44.33it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 44.33it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 44.33it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 44.33it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 44.33it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.01 GB):   3%|▎         | 2/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.24it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.62it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.98 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.34it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.34it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.34it/s]Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.34it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.34it/s]

    Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.34it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=768 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=512 avail_mem=70.94 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.70it/s]Capturing num tokens (num_tokens=512 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=416 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=384 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.00it/s]

    Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=320 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=240 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.84it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.84it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.84it/s]Capturing num tokens (num_tokens=176 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.84it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.84it/s]

    Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.84it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s] Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]

    Capturing num tokens (num_tokens=20 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=20 avail_mem=70.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.17it/s]Capturing num tokens (num_tokens=16 avail_mem=70.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.17it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.17it/s]Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.17it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.17it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 41.69it/s]


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
    Generated text:  Emily, and I'm a software developer from Seattle, WA. I have 10+ years of experience in software development. My specialties include, but are not limited to: web development, mobile development, server-side programming, JavaScript, and backend/DBA skills.
    I have a passion for helping organizations navigate the complexities of software development, with a particular focus on cloud computing and automation. I have a deep understanding of data management, SQL, and cloud infrastructure, and I'm skilled at troubleshooting and resolving technical issues.
    I'm also an avid amateur musician, and I have performed with local bands and taken guitar lessons. I'm currently
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to become more environmentally conscious. He decides to use only 10% of his weekly budget for solar energy. He currently uses 1000 units of energy each week, including 150 units from coal. How many units of energy does he need to reduce by using only 10% of his weekly budget for solar energy?
    To determine how many units of energy the president needs to reduce by using only 10% of his weekly budget for solar energy, we can follow these steps:
    
    1. Calculate the total weekly budget the president currently uses.
    2. Determine the amount of energy he would use if
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Rome
    C. Vienna
    D. London
    答案:
    
    A
    
    下列哪种情形下，可以将公文的标题作为标题正文内容的一部分，单独成页，单面排印____
    A. 整个公文文稿
    B. 公文的主送机关
    C. 公文的标题
    D. 公文的附件名称
    答案:
    
    A
    
    同一个人在不同时期可以有多个实际姓名，这是指____。
    A. 信息的多值性
    B. 信息的互换性
    C. 信息
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and for better or worse, it’s at the heart of many of our personal lives. From personal assistants like Siri, Alexa, and Google Assistant, to the latest technology like self-driving cars and virtual assistants, AI is taking over and changing the world.
    But is it safe for everyone? Is it accessible to everyone? Will AI ever make us want to chat with dogs, watch anime, or even make friends with other humans? And what’s the ethical concerns surrounding AI? This course will introduce you to the technical, ethical, and social aspects of AI, and will explore the major issues surrounding AI that affect our personal lives,


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and art, and is a major tourist destination. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. It is the capital of France and a major economic and political center in Europe. The city is home to many important institutions and organizations, including the French Academy of Sciences and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we can expect to see more automation and artificial intelligence in various industries. This could lead to the development of new jobs, but also create new opportunities for people to work with AI systems.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there will be a greater need for security measures to protect user data. This could lead to the development of new privacy and security standards,
    


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
    Generated text:  [Your Name], a [Your Role] who has been working for [Your Company] for [Your Duration]. I'm a [Your Age], [Your Gender], [Your Nationality] from [Your Location]. I have [Your Key Skills], [Your Interests], and [Your Areas of Expertise] in my field. What can you tell me about yourself? I'm confident and optimistic, able to adapt quickly, and always seeking to learn. I have a passion for [Your Hobby/Interest/Project] that I find incredibly fulfilling. Thank you for considering me for a potential role. What are your current positions or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the world's 16th-largest city and a UNESCO World Heritage site. The city is known for its historic architecture, vibrant nightlife, and luxurious dining options, as well as its importance as a hub for international politics, culture, and commerce. The city is also home to numerous museums, art galleries, and theaters, and has a rich history dating back over 2, 000 years. With its numerous landmarks, including the Eiffel Tower and the Louvre Museum, Paris is a city that is at once unique and universally recognized. The French language is also an official language of the European Union and the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and dynamic, with many potential trends shaping its direction. Some potential future trends include:
    
    1. Autonomous vehicles: AI is increasingly being used to develop self-driving cars, with many companies developing advanced AI algorithms for collision detection, traffic management, and more. This technology could lead to significant changes in transportation, reducing the risk of accidents and improving safety.
    
    2. Personalized medicine: AI could revolutionize the field of medicine by enabling more accurate diagnoses and treatments, as well as personalized treatment plans based on an individual's genetic makeup and medical history.
    
    3. Smart cities: AI could be used to develop smart cities that are more efficient, sustainable


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

     am

     a

    /an

     [

    Title

    ].

     I

     am

     a

    /an

     [

    Age

    ],

     [

    Gender

    ]

     [

    Race

    ].

     I

     have

     [

    Name

    ]

     on

     my

     phone

     and

     [

    Name

    ]

     on

     my

     keyboard

    .

     I

     am

     a

    /an

     [

    Occup

    ation

    ].

     I

     like

     [

    Favorite

     Thing

    ]

     and

     [

    Favorite

     Food

    ].

     I

     also

     love

     [

    Favorite

     Book

    ]

     and

     [

    Favorite

     Music

    ].

     I

     am

     an

     [

    interest

    ]

     person

    .

     I

     like

     to

     [

    Favorite

     Hobby

    /

    Activity

    ].

     I

     am

     always

     looking

     for

     [

    What

     I

    'm

     Looking

     For

    ].

     I

    'm

     always

     trying

     to

     [

    What

     I

    'm

     Trying

     To

     Do

    ].

     I

     hope

     to

     [

    What

     I

     Want

     To

     Do

     In

     The

     Future

    ].


    My

     name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     in

     the

     heart

     of

     the

     Mediterranean

     coast

    .

     It

     is

     a

     significant

     center

     for

     government

    ,

     culture

    ,

     arts

    ,

     science

    ,

     and

     commerce

    .

     The

     city

     hosts

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     famous

     landmarks

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     diverse

     culture

    ,

     and

     beautiful

     architecture

    .

     With

     its

     bustling

     streets

    ,

     picturesque

     parks

    ,

     and

     vibrant

     nightlife

    ,

     Paris

     is

     a

     popular

     tourist

     destination

     worldwide

    .

     As

     the

     largest

     and

     oldest

     city

     in

     Europe

    ,

     Paris

     is

     a

     major

     hub

     of

     global

     culture

     and

     politics

    .

     Its

     unique

     blend

     of

     historical

     and

     modern

     elements

     makes

     it

     a

     fascinating

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     has

     the

     potential

     to

     revolution

    ize

     virtually

     every

     industry

    .

     Here

     are

     some

     possible

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

     coming

     years

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

     can

     be

     used

     to

     improve

     the

     accuracy

     and

     speed

     of

     medical

     diagnosis

    ,

     treatment

    ,

     and

     patient

     care

    .

     For

     example

    ,

     AI

    -powered

     diagnostic

     tools

     can

     identify

     abnormalities

     that

     might

     have

     been

     missed

     by

     human

     doctors

    ,

     and

     can

     even

     assist

     with

     patient

     management

     and

     follow

    -up

     care

    .

     AI

    -powered

     drug

     discovery

     and

     development

     can

     also

     lead

     to

     the

     discovery

     of

     new

     treatments

     for

     diseases

     and

     conditions

     that

     were

     previously

     unt

    reat

    able

    .
    


    2

    .

     Integration

     of

     AI

     with

     other

     technologies

    :

     AI

     is

     already

    



```python
llm.shutdown()
```
