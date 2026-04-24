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
    [2026-04-24 00:56:31] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.04it/s]


    2026-04-24 00:56:36,450 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 00:56:36] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.63it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.63it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.63it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.63it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.63it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.63it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.63it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.63it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.76it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.76it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.76it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.76it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.76it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.76it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.76it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.76it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:03<00:01, 18.76it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:03<00:01, 18.76it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 27.95it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 27.95it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 27.95it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 27.95it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 27.95it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 27.95it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 27.95it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 27.95it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:03<00:00, 27.95it/s]

    Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:03<00:00, 27.95it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 37.15it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 37.15it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 37.15it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 37.15it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 37.15it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 37.15it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 37.15it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:03<00:00, 37.15it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:03<00:00, 37.15it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 44.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.48 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.45 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.45 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.45 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.45 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.45 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.44 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.44 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.44 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.44 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.44 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.05it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.44 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.43 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.43 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.05it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=118.43 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.42 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.42 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.42 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.42 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.41 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.41 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.41 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.41 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.40 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.38 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.22it/s]

    Capturing num tokens (num_tokens=960 avail_mem=118.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.22it/s] Capturing num tokens (num_tokens=960 avail_mem=118.39 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.05it/s]Capturing num tokens (num_tokens=896 avail_mem=118.39 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.05it/s]Capturing num tokens (num_tokens=832 avail_mem=118.39 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.05it/s]Capturing num tokens (num_tokens=768 avail_mem=118.38 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.05it/s]Capturing num tokens (num_tokens=704 avail_mem=118.38 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.05it/s]Capturing num tokens (num_tokens=640 avail_mem=118.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.05it/s]Capturing num tokens (num_tokens=640 avail_mem=118.35 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=576 avail_mem=118.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=512 avail_mem=118.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.50it/s]

    Capturing num tokens (num_tokens=480 avail_mem=118.35 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=448 avail_mem=118.35 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=448 avail_mem=118.35 GB):  53%|█████▎    | 31/58 [00:00<00:00, 35.65it/s]Capturing num tokens (num_tokens=416 avail_mem=118.34 GB):  53%|█████▎    | 31/58 [00:00<00:00, 35.65it/s]Capturing num tokens (num_tokens=384 avail_mem=118.34 GB):  53%|█████▎    | 31/58 [00:00<00:00, 35.65it/s]Capturing num tokens (num_tokens=352 avail_mem=118.34 GB):  53%|█████▎    | 31/58 [00:00<00:00, 35.65it/s]Capturing num tokens (num_tokens=320 avail_mem=118.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=288 avail_mem=118.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=288 avail_mem=118.33 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=256 avail_mem=118.33 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=240 avail_mem=118.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.83it/s]

    Capturing num tokens (num_tokens=224 avail_mem=118.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=208 avail_mem=118.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=192 avail_mem=118.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=192 avail_mem=118.32 GB):  71%|███████   | 41/58 [00:01<00:00, 39.55it/s]Capturing num tokens (num_tokens=176 avail_mem=118.31 GB):  71%|███████   | 41/58 [00:01<00:00, 39.55it/s]Capturing num tokens (num_tokens=160 avail_mem=118.31 GB):  71%|███████   | 41/58 [00:01<00:00, 39.55it/s]Capturing num tokens (num_tokens=144 avail_mem=118.30 GB):  71%|███████   | 41/58 [00:01<00:00, 39.55it/s]Capturing num tokens (num_tokens=128 avail_mem=118.30 GB):  71%|███████   | 41/58 [00:01<00:00, 39.55it/s]Capturing num tokens (num_tokens=112 avail_mem=118.30 GB):  71%|███████   | 41/58 [00:01<00:00, 39.55it/s]Capturing num tokens (num_tokens=112 avail_mem=118.30 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.51it/s]Capturing num tokens (num_tokens=96 avail_mem=118.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.51it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=118.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.51it/s]Capturing num tokens (num_tokens=64 avail_mem=118.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.51it/s]Capturing num tokens (num_tokens=48 avail_mem=118.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.51it/s]Capturing num tokens (num_tokens=32 avail_mem=118.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.51it/s]Capturing num tokens (num_tokens=32 avail_mem=118.28 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=28 avail_mem=118.28 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=24 avail_mem=118.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=20 avail_mem=118.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=16 avail_mem=118.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=12 avail_mem=118.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.83it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.27 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=8 avail_mem=118.26 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.37it/s] Capturing num tokens (num_tokens=4 avail_mem=118.26 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=4 avail_mem=118.26 GB): 100%|██████████| 58/58 [00:01<00:00, 36.76it/s]


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
    Generated text:  GARY HALL
    In my early childhood, I was the black-hearted kid who couldn’t wait to go to school. I was the one who got all the attention and received all the praise. I was the one who was always the one to take on the leadership role in class and was the one who always thought that everyone was going to be my friend. I always thought I was the best. I thought that I was the one who was going to do everything right. I was the one who always did everything perfectly. I was the one who was always the one to be the leader of the group. I was always the one
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful leader of the nation. He is the highest executive authority of the government. He is the leader of the country. As the leader of the country, the president exercises all the power and authority of the government. The president’s power, authority, and influence is so great that he can change the government of the country. The president is the person who appoints the other government officers and the head of the government. He also the president makes the laws of the country and appoints judges. The president also makes speeches and writes letters. The president’s power of the presidency has changed since the time of the first president. He is
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. London
    B. Paris
    C. Moscow
    D. New York
    Answer:
    
    B
    
    In the following reactions, which one is a displacement reaction? 
    A. S + O2 → SO2
    B. CaO + H2O → Ca(OH)2
    C. 2KMnO4 + 16HCl → 2KCl + 2MnCl2 + 5Cl2 ↑ + 8H2O
    D. Al + 3HCl → AlCl3 + H2↑
    Answer:
    
    C
    
    Which of the following statements about the
    ===============================
    Prompt: The future of AI is
    Generated text:  here. And it's not for the better. Over the last few years, AI has helped transform the world as we know it. But as the technology advances and the data it consumes grows, so does the environmental impact of AI.
    It's not just the environmental damage that comes from the exponential growth of AI. It's also the loss of jobs. The shortage of AI workers and the poor wages they earn has left many people feeling the brunt of the increase in technology.
    However, the good news is that there are steps we can all take to reduce our impact on the environment. One of the most important things to do is to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm always looking for new ways to improve myself and make the world a better place. What do you enjoy doing in your free time? I enjoy reading, playing sports, and spending time with my family. What's your favorite hobby? I love to travel and explore new places
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture. The city is known for its beautiful architecture, vibrant nightlife, and annual festivals such as the Eiffel Tower and the Louvre Museum. Paris is also a major center for art, music, and literature, and is home to many famous landmarks and museums. The city is a popular tourist destination and a major economic hub in France. It is also a major center for politics, with the French Parliament located in the city. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and preferences. This could lead to more personalized and efficient solutions to complex problems.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more transparent and accountable AI systems, as well as more robust safeguards against bias and discrimination.
    
    3.
    


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
    Generated text:  [Name], and I'm a [Occupation] who has been [Number of Years] years in this profession. I enjoy [What I enjoy doing], and I strive to [What I try to achieve], every day. I'm [Age], and I've always been [What I was born into or started from]. I have a [Favorite Hobby or Family Activity] that I always strive to [What I have learned or gained from it]. I'm a [What my profession is like to a character] who is always [What a characteristic I have that differentiates me from others]. I hope to have [A short goal
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and the second-largest city in the world, with a population of over 2.1 million. The city is known for its rich history, world-class museums, and vibrant arts scene. Paris is also home to the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. With its medieval architecture, romantic ambiance, and thriving economy, Paris is a must-visit destination for any traveler visiting France. 
    
    France's capital city Paris is a historical and cultural gem, and is widely recognized as the capital of the European Union. It is also known as the "City
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by a number of trends that are expected to continue and even grow in importance. Some of the potential future trends in AI include:
    
    1. Increased use of AI in healthcare: As AI technology continues to improve, it is expected to have a significant impact on healthcare. AI-powered diagnostic tools, personalized medicine, and automated patient care will become more common in healthcare.
    
    2. Integration of AI with other technologies: The integration of AI with other technologies, such as the Internet of Things (IoT), will continue to grow in importance. AI-powered smart home devices, self-driving cars, and virtual assistants are all examples of how


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

    /an

     [

    Character

    ]

     who

     has

     been

     [

    Character

    's

     Character

    istic

    ]

     for

     [

    Number

     of

     Years

    ].

     I

    'm

     always

     looking

     for

     the

     next

     big

     thing

     to

     pursue

    ,

     and

     I

    'm

     always

     on

     the

     lookout

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     learn

    .

     I

    'm

     a

     [

    job

     title

    ]

     who

     is

     always

     looking

     for

     ways

     to

     make

     the

     world

     a

     better

     place

    .

     I

    'm

     excited

     to

     meet

     new

     people

     and

     try

     new

     things

    ,

     and

     I

    'm

     always

     up

     for

     a

     challenge

    .

     What

    's

     your

     name

     and

     what

     do

     you

     do

    ?

     [

    Name

    ]

    !

     Welcome

     to

     the

     world

     of

     [

    Character

    's

     Character

    istic

    ]

     and

     [

    Name

    ],

     a

     place

    
    
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

    .

     Paris

     is

     a

     city

     located

     on

     the

     island

     of

     France

     and

     is

     the

     most

     populous

     city

     in

     France

    .

     It

     is

     the

     seat

     of

     government

     and

     culture

     and

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     the

     home

     of

     the

     French

     Riv

    iera

     and

     is

     a

     major

     tourist

     destination

     in

     Europe

    .

     The

     city

     is

     often

     referred

     to

     as

     "

    the

     City

     of

     Light

    "

     due

     to

     its

     elegant

     and

     bustling

     atmosphere

    .

     It

     is

     home

     to

     over

     

    3

     million

     residents

     and

     

    1

    2

     million

     visitors

     annually

    .

     Paris

     is

     also

     a

     hub

     for

     finance

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     speculative

    ,

     but

     there

     are

     several

     trends

     that

     are

     likely

     to

     shape

     its

     development

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     privacy

     and

     data

     protection

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     increased

     need

     for

     security

     measures

     to

     protect

     personal

     data

     and

     privacy

    .

     As

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     privacy

    -focused

     AI

     that

     priorit

    izes

     user

     safety

     and

     security

    .
    


    2

    .

     Automation

     and

     automation

    :

     The

     automation

     of

     manual

     tasks

     is

     becoming

     more

     prevalent

     in

     various

     industries

    ,

     from

     manufacturing

     and

     transportation

     to

     healthcare

     and

     finance

    .

     AI

    -powered

     automation

     could

     make

     our

     jobs

     safer

    ,

     more

     efficient

    ,

     and

     more

     secure

    .
    


    3

    .

     Self

    -driving

     cars

    :

     Self

    -driving

    



```python
llm.shutdown()
```
