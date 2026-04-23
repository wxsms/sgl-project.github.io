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
    [2026-04-23 17:25:55] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.31it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.31it/s]


    2026-04-23 17:26:11,350 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 17:26:11] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 13.36it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 13.36it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 13.36it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 13.36it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.36it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.36it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.36it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 13.36it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 13.36it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 13.36it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 30.23it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 30.23it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 30.23it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 30.23it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 30.23it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 30.23it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 30.23it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 30.23it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 30.23it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 30.23it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:03<00:00, 30.23it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 40.75it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 40.75it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 40.75it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 40.75it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 40.75it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 40.75it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 40.75it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 40.75it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 40.75it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 40.75it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 40.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.47 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.47 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.47 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.46 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.46 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.46 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.46 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.46 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.46 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.46 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.46 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.45 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.45 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.45 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.44 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=73.44 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.44 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.44 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.43 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.43 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.43 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.42 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.42 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.42 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.40 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s]Capturing num tokens (num_tokens=960 avail_mem=73.41 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=73.41 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s]Capturing num tokens (num_tokens=832 avail_mem=73.41 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.07it/s]Capturing num tokens (num_tokens=832 avail_mem=73.41 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.53it/s]Capturing num tokens (num_tokens=768 avail_mem=73.40 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.53it/s]Capturing num tokens (num_tokens=704 avail_mem=73.40 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.53it/s]Capturing num tokens (num_tokens=640 avail_mem=73.40 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.53it/s]Capturing num tokens (num_tokens=576 avail_mem=73.39 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.53it/s]Capturing num tokens (num_tokens=512 avail_mem=73.39 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.53it/s]Capturing num tokens (num_tokens=512 avail_mem=73.39 GB):  50%|█████     | 29/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=480 avail_mem=73.40 GB):  50%|█████     | 29/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=448 avail_mem=73.40 GB):  50%|█████     | 29/58 [00:00<00:00, 40.68it/s]

    Capturing num tokens (num_tokens=416 avail_mem=73.40 GB):  50%|█████     | 29/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=384 avail_mem=73.40 GB):  50%|█████     | 29/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=352 avail_mem=73.39 GB):  50%|█████     | 29/58 [00:00<00:00, 40.68it/s]Capturing num tokens (num_tokens=352 avail_mem=73.39 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.39it/s]Capturing num tokens (num_tokens=320 avail_mem=73.39 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.39it/s]Capturing num tokens (num_tokens=288 avail_mem=73.38 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.39it/s]Capturing num tokens (num_tokens=256 avail_mem=73.38 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.39it/s]Capturing num tokens (num_tokens=240 avail_mem=73.38 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.39it/s]Capturing num tokens (num_tokens=224 avail_mem=73.37 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.39it/s]Capturing num tokens (num_tokens=224 avail_mem=73.37 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=208 avail_mem=73.37 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.01it/s]

    Capturing num tokens (num_tokens=192 avail_mem=73.37 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=176 avail_mem=73.36 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=160 avail_mem=73.36 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=144 avail_mem=73.36 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=144 avail_mem=73.36 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=128 avail_mem=73.36 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=112 avail_mem=73.35 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=96 avail_mem=73.35 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.59it/s] Capturing num tokens (num_tokens=80 avail_mem=73.35 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=64 avail_mem=73.34 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=64 avail_mem=73.34 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=48 avail_mem=73.34 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.41it/s]

    Capturing num tokens (num_tokens=32 avail_mem=73.34 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=28 avail_mem=73.33 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=24 avail_mem=73.33 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=20 avail_mem=73.32 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.41it/s]Capturing num tokens (num_tokens=20 avail_mem=73.32 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=16 avail_mem=73.32 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=12 avail_mem=73.32 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=8 avail_mem=73.32 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.65it/s] Capturing num tokens (num_tokens=4 avail_mem=73.31 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=4 avail_mem=73.31 GB): 100%|██████████| 58/58 [00:01<00:00, 39.94it/s]


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
    Generated text:  Nick and I am a medical student from the United States. I have been reading a lot of books on the topic of artificial intelligence, and I have seen that it has the potential to revolutionize the world. However, I am interested in learning more about the ethical implications of AI. What are some of the key ethical issues that arise in the use of AI, and how can we address them? Additionally, what can we do to ensure that AI is used responsibly and ethically? As a medical student, I am particularly concerned about the impact of AI on the human body and its ability to process and interpret medical data. How can I
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to determine how many U.S. presidents have served as the head of a federal agency. In what year did the first president to serve as head of a federal agency occur? To determine the year in which the first president to serve as head of a federal agency was sought by the president, we need to list the presidents of the United States and identify the one who held this position from the earliest to the most recent. Here are the presidents in chronological order of their earliest service to their most recent service:
    
    1. George Washington (1789-1797)
    2. John Adams (1797-1
    ===============================
    Prompt: The capital of France is
    Generated text:  located in which region of the country? Paris is located in the Île de France, a part of the Île-de-France region in northwestern France.
    The answer is: Île de France.
    ===============================
    Prompt: The future of AI is
    Generated text:  definitely in the hands of humans, but a study by the MIT Media Lab has raised questions about the ethical implications of the potential dangers that could be caused by human AI. A report published in the Journal of the ACM by the Media Lab discusses the issues surrounding the use of AI in areas ranging from medical diagnosis to investment in stocks.
    The report suggests that AI has the ability to mislead humans, making decisions based on incomplete or inaccurate data. In addition, AI can cause harm to people by affecting their jobs or even stealing their wealth.
    The report warns that as AI continues to improve and become more capable, it will be essential to consider the


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I have always been passionate about [Your Passion], and I am always looking for ways to [Your Goal]. I am a [Your Character Trait] who is always [Your Character Quality]. I am [Your Character Name] and I am excited to meet you. [Your Character Name] is always ready to learn and grow, and I am always eager to share my knowledge and experiences with you. I am a [Your Character Name] who is always [Your Character Quality]. I am [Your Character Name] and I am always [Your Character Trait].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the world. Paris is also home to many famous French artists, writers, and musicians. The city is known for its cuisine, including its famous Parisian cuisine, and its fashion industry. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability. AI developers will need to be more mindful of the potential impact of their technology on society.
    
    2. Integration with human decision-making: AI is likely to become more integrated with human decision-making in the future. This will involve the use of AI to assist with decision-making processes, such as in healthcare
    


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
    Generated text:  [Name], and I am a [职业] with a passion for [职内的专业] and a keen interest in [职内的领域] of knowledge. I've always been fascinated by the [职内的领域] because [为什么我被吸引] and I'm always seeking to learn more about it. I'm [在业余时间] my passion for my hobby and I'm always looking for opportunities to share my knowledge with others. Thank you for having me. [结束语]
    职业: 研发工程师
    专业: 机器学习
    领域: 深度学习
    业余时间: 与朋友和家人
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the city of lights. It is the largest and most populous city in France, and is located in the western part of the country, on the Seine River. It is considered to be the cultural, artistic, and economic heart of France, and is known for its rich history, beautiful architecture, and numerous famous landmarks, including the Louvre Museum and the Eiffel Tower. Paris is also an important center of international politics, with many of the world’s most influential companies and institutions headquartered there. It is home to many important French schools, such as École Polytechnique and the École des Be
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, and there are several areas where we can expect significant advancements and changes. Here are some possible future trends in artificial intelligence:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient care and make diagnoses faster and more accurate. We may see even more use of AI in the future as the technology continues to advance.
    
    2. Personalized medicine: AI is already being used to help doctors diagnose and treat patients more accurately. We may see even more use of AI in the future as the technology continues to improve and become more personalized.
    
    3. Autonomous vehicles: AI is already being used in autonomous vehicles


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

     am

     a

     [

    Name

    ]

    !


    Nice

     to

     meet

     you

    !

     I

    'm

     a

     [

    Name

    ]

     who

     has

     been

     working

     as

     a

     [

    Name

    ]

     for

     [

    Name

    ]

     for

     [

    Name

    ]

     years

    .

     I

    'm

     very

     dedicated

     and

     always

     willing

     to

     take

     on

     new

     challenges

    .

     What

     can

     you

     tell

     me

     about

     your

     work

     experience

    ?

     Hey

    ,

     thanks

     for

     chatting

     with

     me

    !

     What

    's

     your

     job

     title

    ?


    Hi

    ,

     I

    'm

     [

    Name

    ],

     a

     [

    Name

    ]

     with

     [

    Name

    ]

     for

     [

    Name

    ]

     years

    .

     I

    've

     been

     working

     in

     [

    Name

    ]

     for

     [

    Name

    ]

     years

     and

     I

    've

     been

     working

     for

     [

    Name

    ]

     for

     [

    Name

    ]

     years

    .

     What

    's

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     largest

     city

     in

     Europe

     and

     is

     known

     for

     its

     rich

     history

    ,

     museums

    ,

     and

     fine

     dining

    .

     It

    's

     also

     the

     oldest

     continuously

     occupied

     capital

     city

     in

     the

     world

    ,

     having

     been

     founded

     in

     

    7

    9

    3

     AD

     by

     Char

    lem

    agne

    .

     The

     city

     has

     a

     diverse

     population

     of

     over

     

    3

     million

     people

     and

     is

     the

     most

     visited

     city

     in

     the

     world

    ,

     attracting

     millions

     of

     tourists

     every

     year

    .

     Paris

     is

     a

     symbol

     of

     France

     and

     a

     popular

     tourist

     destination

    ,

     and

     is

     a

     major

     economic

     and

     cultural

     center

    .

     Additionally

    ,

     it

     is

     considered

     to

     be

     a

     global

     cultural

     and

     intellectual

     hub

    ,

     attracting

     artists

    ,

     writers

    ,

     musicians

    ,

     and

     other

     intellectuals

     from

     around

     the

     world

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     technological

     advancement

     and

     diverse

     applications

     across

     various

     sectors

    .

     Some

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     emphasis

     on

     ethical

     considerations

    :

     As

     AI

     continues

     to

     evolve

    ,

     there

     will

     be

     increasing

     pressure

     to

     ensure

     that

     its

     development

     and

     deployment

     are

     ethical

     and

     respectful

     of

     human

     rights

    ,

     privacy

    ,

     and

     other

     fundamental

     values

    .
    


    2

    .

     Greater

     focus

     on

     AI

     transparency

    :

     With

     the

     rise

     of

     AI

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     ensuring

     that

     AI

     systems

     are

     transparent

     and

     can

     be

     easily

     understood

     and

     explained

     to

     humans

    .
    


    3

    .

     AI

     integration

     with

     other

     technologies

    :

     AI

     is

     increasingly

     being

     integrated

     with

     other

     technologies

     such

     as

     sensors

    ,

     drones

    ,

     and

     IoT

     devices

    ,

     creating

    



```python
llm.shutdown()
```
