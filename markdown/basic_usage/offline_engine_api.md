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
    [2026-04-21 06:19:42] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.90it/s]


    2026-04-21 06:19:47,103 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 06:19:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.77it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.77it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.77it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.77it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.77it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.77it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.77it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.77it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.77it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 19.95it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 19.95it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 19.95it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 19.95it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 19.95it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 19.95it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 19.95it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 19.95it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 19.95it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:03<00:01, 19.95it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 29.09it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:03<00:00, 29.09it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 38.16it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:03<00:00, 38.16it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 47.15it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 47.15it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 47.15it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 47.15it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 47.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.09 GB):   3%|▎         | 2/58 [00:00<00:03, 17.77it/s]Capturing num tokens (num_tokens=7168 avail_mem=117.09 GB):   3%|▎         | 2/58 [00:00<00:03, 17.77it/s]Capturing num tokens (num_tokens=6656 avail_mem=117.09 GB):   3%|▎         | 2/58 [00:00<00:03, 17.77it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=117.09 GB):   3%|▎         | 2/58 [00:00<00:03, 17.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=117.09 GB):   9%|▊         | 5/58 [00:00<00:02, 20.18it/s]Capturing num tokens (num_tokens=5632 avail_mem=117.08 GB):   9%|▊         | 5/58 [00:00<00:02, 20.18it/s]Capturing num tokens (num_tokens=5120 avail_mem=117.08 GB):   9%|▊         | 5/58 [00:00<00:02, 20.18it/s]Capturing num tokens (num_tokens=4608 avail_mem=117.08 GB):   9%|▊         | 5/58 [00:00<00:02, 20.18it/s]Capturing num tokens (num_tokens=4608 avail_mem=117.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=117.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=117.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=3584 avail_mem=117.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=117.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=3328 avail_mem=117.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=117.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.92it/s]Capturing num tokens (num_tokens=2816 avail_mem=117.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.92it/s]Capturing num tokens (num_tokens=2560 avail_mem=117.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.92it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.92it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.05 GB):  21%|██        | 12/58 [00:00<00:01, 28.92it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.57it/s]Capturing num tokens (num_tokens=1792 avail_mem=117.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=117.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=117.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.57it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=117.02 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.57it/s]Capturing num tokens (num_tokens=960 avail_mem=117.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.57it/s] Capturing num tokens (num_tokens=960 avail_mem=117.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=896 avail_mem=117.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=832 avail_mem=117.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=768 avail_mem=117.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=704 avail_mem=117.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=640 avail_mem=117.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=640 avail_mem=117.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.70it/s]Capturing num tokens (num_tokens=576 avail_mem=117.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.70it/s]Capturing num tokens (num_tokens=512 avail_mem=117.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.70it/s]

    Capturing num tokens (num_tokens=480 avail_mem=117.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.70it/s]Capturing num tokens (num_tokens=448 avail_mem=117.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.70it/s]Capturing num tokens (num_tokens=416 avail_mem=117.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.70it/s]Capturing num tokens (num_tokens=416 avail_mem=117.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.31it/s]Capturing num tokens (num_tokens=384 avail_mem=117.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.31it/s]Capturing num tokens (num_tokens=352 avail_mem=117.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.31it/s]Capturing num tokens (num_tokens=320 avail_mem=117.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.31it/s]Capturing num tokens (num_tokens=288 avail_mem=117.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.31it/s]Capturing num tokens (num_tokens=288 avail_mem=117.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=256 avail_mem=117.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=240 avail_mem=117.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.82it/s]

    Capturing num tokens (num_tokens=224 avail_mem=117.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=208 avail_mem=116.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=192 avail_mem=116.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=192 avail_mem=116.99 GB):  71%|███████   | 41/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=176 avail_mem=116.99 GB):  71%|███████   | 41/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=160 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=144 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=128 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=112 avail_mem=116.98 GB):  71%|███████   | 41/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=112 avail_mem=116.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.27it/s]Capturing num tokens (num_tokens=96 avail_mem=116.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.27it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=116.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.27it/s]Capturing num tokens (num_tokens=64 avail_mem=116.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.27it/s]Capturing num tokens (num_tokens=48 avail_mem=116.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.27it/s]Capturing num tokens (num_tokens=32 avail_mem=116.95 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.27it/s]Capturing num tokens (num_tokens=32 avail_mem=116.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=28 avail_mem=116.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=24 avail_mem=116.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=20 avail_mem=116.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=16 avail_mem=116.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=12 avail_mem=116.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.60it/s]

    Capturing num tokens (num_tokens=12 avail_mem=116.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.54it/s]Capturing num tokens (num_tokens=8 avail_mem=116.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.54it/s] Capturing num tokens (num_tokens=4 avail_mem=116.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.54it/s]Capturing num tokens (num_tokens=4 avail_mem=116.93 GB): 100%|██████████| 58/58 [00:01<00:00, 36.91it/s]


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
    Generated text:  Claire and I am 15 years old. What are you doing for fun? : eHow
    Hello, my name is Claire and I am 15 years old. What are you doing for fun?
    It's great to hear that you're interested in your own personal interests and hobbies. Do you have any specific hobbies or interests that you'd like to share with me? I'd be happy to chat and learn more about your interests and experiences! 😊😊😊
    
    My name is Claire and I am 15 years old. I enjoy going to the park and playing sports. What are some of your favorite sports?
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  a popular role model, and young people aspire to follow in his footsteps. However, as the 2016 United States presidential election approached, the challenges faced by young people were highlighted, emphasizing the importance of making better choices when it comes to politics. This suggests that:
    A. The president's policies will benefit young people
    B. The president will take responsibility
    C. Young people should be more open-minded
    D. Young people should be more influential
    Answer:
    
    C
    
    How many different arrangements are there for arranging 2 distinct objects (A and B) into 3 distinguishable slots (a, b, and c
    ===============================
    Prompt: The capital of France is
    Generated text:  _____.
    A. Paris
    B. Toulouse
    C. Marseille
    D. Lyon
    Answer: A
    
    The school of thought that emphasizes the use of functionalist theories to explain the origin, evolution, and development of art and design is ____.
    A. Classicism
    B. Romanticism
    C. Postmodernism
    D. Modernism
    Answer: D
    
    Which of the following statements about the school of thought that emphasizes the use of functionalist theories to explain the origin, evolution, and development of art and design is __________.
    A. The origin, evolution, and development of art and design are subject to historical
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it's being transformed by the latest technological advancements.
    Since 2012, the European Union has been pushing the world towards a single, unified AI strategy, launching initiatives such as the European Union AI Roadmap (EU AI Roadmap) and the AI strategy document, which is a blueprint for the future of AI. Since 2015, the EU has deployed the Artificial Intelligence Roadmap for the Digital Single Market, which enables the coordinated development of AI infrastructure in Europe.
    In the coming years, the EU is aiming to implement the Union's AI strategy to foster a strong European AI sector, and improve the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a few key points about yourself, such as your education, experience, skills, or interests]. What do you enjoy doing in your free time? I enjoy [insert a few hobbies or interests, such as reading, hiking, or playing music]. What's your favorite book or movie? I love [insert a few favorite books or movies, such as [name of book/movie], [name of movie], or [name of song
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Academy of Sciences. Paris is a bustling metropolis with a rich cultural heritage and is a major economic and political center in Europe. Its history dates back to the Roman Empire and is known for its rich history and cultural heritage. The city is also home to many famous French artists, writers, and musicians. Paris is a city that is both a cultural and historical center, and it continues to be a major hub for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy. AI developers will need to be more mindful of the potential consequences of their work and work to ensure that it is used in a responsible and ethical manner.
    
    2. Greater integration with other technologies: AI is already being integrated into
    


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
    Generated text:  [insert character's name here]. I'm a [insert job title here] with over [insert number of years in the industry here] years of experience in [insert relevant field here]. I'm a [insert personality trait here, such as "ambitious", "productive", "team-oriented", etc.]. I'm a team player, approachable, and work well in a fast-paced environment. I'm proficient in [insert relevant software or tools here], and I'm always looking to learn new skills. I'm dedicated to always improving my knowledge and skills. I'm passionate about [insert a personal statement about your personal interests
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known as the "City of Love" due to the large number of romantic buildings in the city.
    
    **Paris City Facts:**
    
    1. **Latitude:** Approximately 48.8566° N
    2. **Longitude:** Approximately 2.2944° E
    3. **Elevation:** 313 meters above sea level
    4. **Population:** Approximately 2.2 million (as of 2021)
    5. **Total Area:** 202.34 km² (as of 2021)
    6. **Economic Importance
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to evolve and develop in exciting and unpredictable ways. Here are some of the possible trends that are currently predicted to shape the future of AI:
    
    1. Integration with human cognition: The integration of AI with human cognition could lead to new forms of artificial intelligence that are more complex, adaptive, and self-aware. This could include AI systems that can learn and adapt to new situations and environments, as well as systems that are able to think and reason like humans.
    
    2. Personalization and relevance: AI is already making efforts to personalize the way people interact with AI systems and to make them relevant to their individual needs. However, as


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

     [

    职业

    /

    特长

    ]

     with

     a

     passion

     for

     [

    职业

    /

    特长

    ]

    !
    


    As

     someone

     with

     a

     [

    职业

    /

    特长

    ]

     background

    ,

     I

     am

     dedicated

     to

     [

    职业

    /

    特长

    ]

     and

     am

     always

     seeking

     to

     improve

     my

     skills

     and

     knowledge

    .

     Whether

     it

    's

     [

    职业

    /

    特长

    ]

     or

     [

    其他

    职业

    /

    特长

    ],

     I

     am

     always

     here

     to

     help

     and

     to

     learn

    .
    


    In

     my

     free

     time

    ,

     I

     enjoy

     [

    职业

    /

    特长

    ]

     and

     other

     hobbies

     that

     I

     find

     enjoyable

    .

     I

     am

     constantly

     seeking

     out

     new

     experiences

     and

     learning

     new

     things

    ,

     and

     I

     believe

     that

     these

     experiences

     can

     help

     me

     grow

     and

     develop

     as

     a

     person

    .
    


    I

     am

    
    
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

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     vibrant

     art

     scene

    ,

     and

     historic

     Lou

    vre

     Museum

    .

     
    


    Sum

    mar

    ize

     the

     events

     leading

     up

     to

     the

     French

     Revolution

    ,

     focusing

     on

     the

     role

     of

     the

     Committee

     of

     Public

     Safety

    .

     Provide

     a

     brief

     summary

     of

     the

     French

     government

    's

     response

     to

     the

     revolutionary

     government

    's

     activities

    .

     Discuss

     the

     outcome

     of

     the

     Committee

     of

     Public

     Safety

    's

     actions

    ,

     including

     its

     impact

     on

     the

     revolution

     and

     the

     establishment

     of

     the

     First

     French

     Republic

    .

     The

     Committee

     of

     Public

     Safety

     was

     formed

     in

     

    1

    7

    8

    9

     and

     was

     tasked

     with

     maintaining

     order

     and

     preventing

     the

     spread

     of

     revolutionary

     ideas

     in

     France

    .

     The

     Committee

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

    ,

     and

     there

     are

     many

     trends

     that

     we

     can

     expect

     to

     see

     in

     the

     coming

     years

    .

     Here

     are

     some

     of

     the

     most

     notable

    :
    


    1

    .

     Improved

     interpret

    ability

     and

     transparency

    :

     As

     AI

     becomes

     more

     complex

     and

     sophisticated

    ,

     it

     is

     becoming

     increasingly

     important

     for

     it

     to

     be

     more

     understandable

     and

     transparent

    .

     This

     means

     that

     we

     will

     see

     more

     AI

     systems

     that

     are

     designed

     to

     be

     more

     human

    -like

     and

     able

     to

     explain

     their

     decisions

     and

     actions

    .
    


    2

    .

     Integration

     of

     AI

     into

     common

     tools

     and

     devices

    :

     The

     integration

     of

     AI

     into

     everyday

     tools

     and

     devices

     will

     continue

     to

     grow

     in

     popularity

    .

     We

     will

     see

     more

     of

     these

     tools

     and

     devices

    ,

     such

     as

     voice

     assistants

    ,

     smart

     home

     systems

    



```python
llm.shutdown()
```
