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
    [2026-04-16 03:06:04] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.83it/s]


    2026-04-16 03:06:10,133 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 03:06:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.85it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.85it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.09it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.09it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.09it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.09it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.09it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.09it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.09it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.09it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.09it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:03<00:01, 20.09it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:03<00:00, 29.35it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 38.58it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:03<00:00, 38.58it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 47.74it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 47.74it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 47.74it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 47.74it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 47.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.51it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.75 GB):   3%|▎         | 2/58 [00:00<00:04, 12.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.43 GB):   3%|▎         | 2/58 [00:00<00:04, 12.68it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=120.15 GB):   3%|▎         | 2/58 [00:00<00:04, 12.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.15 GB):   7%|▋         | 4/58 [00:00<00:04, 12.03it/s]Capturing num tokens (num_tokens=6144 avail_mem=119.76 GB):   7%|▋         | 4/58 [00:00<00:04, 12.03it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.44 GB):   7%|▋         | 4/58 [00:00<00:04, 12.03it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=119.44 GB):   7%|▋         | 4/58 [00:00<00:04, 12.03it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.44 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.44 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.44 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.44 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.43 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.43 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.43 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=119.43 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.23it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=119.43 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.42 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.42 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=119.42 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=119.41 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.31it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=119.41 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.41 GB):  31%|███       | 18/58 [00:00<00:01, 21.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=119.13 GB):  31%|███       | 18/58 [00:00<00:01, 21.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=119.07 GB):  31%|███       | 18/58 [00:00<00:01, 21.13it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  31%|███       | 18/58 [00:01<00:01, 21.13it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  36%|███▌      | 21/58 [00:01<00:02, 18.15it/s]Capturing num tokens (num_tokens=960 avail_mem=118.97 GB):  36%|███▌      | 21/58 [00:01<00:02, 18.15it/s] Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  36%|███▌      | 21/58 [00:01<00:02, 18.15it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  36%|███▌      | 21/58 [00:01<00:02, 18.15it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.22it/s]Capturing num tokens (num_tokens=768 avail_mem=118.96 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.22it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.22it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.22it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.22it/s]

    Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  41%|████▏     | 24/58 [00:01<00:01, 20.22it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  50%|█████     | 29/58 [00:01<00:01, 25.96it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:01<00:01, 25.96it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:01<00:01, 25.96it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:01<00:01, 25.96it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:01<00:01, 25.96it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  50%|█████     | 29/58 [00:01<00:01, 25.96it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.66it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.66it/s]Capturing num tokens (num_tokens=288 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.66it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.66it/s]

    Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.66it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.66it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.19it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.19it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.19it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.19it/s]Capturing num tokens (num_tokens=160 avail_mem=118.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.19it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.19it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.86it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.86it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.86it/s]

    Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.86it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.86it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.86it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.06it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.06it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.06it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.06it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.06it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.06it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.41it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.41it/s] Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:02<00:00, 28.71it/s]


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
    Generated text:  I am a science fiction writer. I am a long-time fan of science fiction, especially space opera, and I want to write a story for a science fiction journal called "The Monstrous Unicorn."
    First of all, I would like to ask you a few questions to get to know you better:
    1. What is your favorite genre of fiction?
    2. What is your favorite science fiction writer?
    3. What is the most important aspect of your favorite science fiction writer?
    4. What is your favorite character from your favorite science fiction writer?
    5. If you had to choose only one science fiction book to read, which one and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the ____.
    A. Cabinet
    B. House of Representatives
    C. Senate
    D. President's Cabinet
    Answer:
    D
    
    The circuit shown in the figure is a voltage divider circuit with resistors R1 and R2, as shown in the figure. The voltages U1 and U2 at this time are respectively (____).
    Answer:
    B
    
    Regarding the "Enterprise Current Situation Sheet", which of the following statements is correct?
    A. It is a schedule for specific departments within an enterprise.
    B. It is a schedule for specific departments within an enterprise, but not for the entire organization.
    C. It
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris  
    B) Lille  
    C) Lyon  
    D) Marseille
    
    To determine the capital of France, we can recall the order of the capitals of France. The capital cities of France are typically arranged in the following order:
    
    1. Paris
    2. London
    3. Madrid
    4. Milan
    5. Rome
    6. Berlin
    7. Brussels
    8. Antwerp
    9. Amsterdam
    10. Lille
    11. Lyon
    12. Marseille
    
    Based on this order, the capital of France is Paris. Therefore, the correct answer is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it's not going to be where we want it to be. It is going to be a place where we want it to be, but there's no way to predict exactly where it's going to be. The technology is moving more quickly than we would like. It's bringing us everything from self-driving cars to the Internet of Things, but it's also bringing a lot of problems.
    AI is capable of problems, but it's also capable of creating solutions. The great thing about AI is that it allows us to think about and solve problems much more efficiently. It allows us to find ways to optimize and make better use


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm a [job title] at [company name] and I'm excited to be here today. I'm a [job title] at [company name] and I'm excited to be here today. I'm a [job title] at [company name] and I'm excited to be here today. I'm a [job title] at [company name] and I'm excited to be here today. I'm a [job title] at [company name] and I'm excited to be here today. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and a vibrant culture. It is located on the Seine River and is the largest city in France by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also famous for its fashion industry, art, and cuisine. Paris is a popular tourist destination and a cultural hub, attracting millions of visitors each year. It is a major economic center and a major center of politics and diplomacy in France. Paris is also home to many international organizations and institutions, including UNESCO
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with applications ranging from personalized medicine to disease diagnosis and treatment.
    
    2. AI in manufacturing: AI is already being used in manufacturing to improve efficiency and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in manufacturing, with applications ranging from predictive maintenance to quality control.
    
    3. AI in finance:
    


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
    Generated text:  Sarah and I am a [insert profession or identity]. How can I best introduce myself to someone new?
    I am Sarah, a [insert character role]. I am [insert character name], a [insert character's occupation]! 🌟
    I am [insert character's profession or identity], and I am here to share my [insert character's hobbies, interests, or experiences]. And if you have any questions or need assistance, feel free to ask! 🛠️
    Please let me know if you're interested in chatting with me more! 🎉
    Warm regards, [insert your name]
    Your friend Sarah �
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city and capital of France. It is located on the Seine River, on the Île de la Cité, and by the River Seine, in the center of the city of Paris. It is the most populous city of the European Union, with a population of approximately 2.2 million people. Paris is also one of the world's most important cultural, economic, and political centers. It is known for its architecture, art, food, and nightlife. It has a rich history, including the Roman and medieval influences of the ancient city of Carthage and the medieval influence of the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but some potential trends that could emerge include:
    
    1. Increased development of AI that can learn from data and adapt to new situations without explicit programming.
    
    2. More advanced AI that can perform tasks that were previously considered impossible, such as human-like creativity, autonomous decision-making, and emotional intelligence.
    
    3. The integration of AI with other technologies, such as the Internet of Things (IoT), to create more interconnected and smarter devices and systems.
    
    4. The development of more ethical AI that can be programmed to follow specific ethical guidelines and principles.
    
    5. The growth of AI-powered autonomous vehicles that can navigate and interact with the world on


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

     name

    ],

     and

     I

    'm

     a

     [

    insert

     occupation

     or

     profession

    ]

     with

     a

     passion

     for

     [

    insert

     something

     related

     to

     your

     occupation

     or

     passion

    ].

     I

    've

     always

     been

     drawn

     to

     the

     beauty

     of

     nature

     and

     am

     always

     looking

     for

     opportunities

     to

     connect

     with

     others

    .

     I

    'm

     always

     excited

     to

     explore

     new

     places

     and

     learn

     new

     things

    ,

     and

     I

    'm

     always

     looking

     for

     a

     new

     challenge

     to

     try

    .

     I

    'm

     passionate

     about

     sharing

     my

     experiences

     and

     knowledge

     with

     others

     and

     I

     enjoy

     helping

     people

     grow

     their

     own

     personal

     growth

    .

     I

    'm

     looking

     forward

     to

     meeting

     you

    !

     [

    insert

     your

     name

    ]

     


    You

     can

     use

     any

     appropriate

     title

     or

     catch

    phrase

     to

     create

     a

     more

     engaging

     self

    -int

    roduction

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     response

     is

     factual

    .

     It

     states

     the

     location

     of

     Paris

     in

     relation

     to

     the

     country

     of

     France

    .

     It

     is

     a

     concise

     and

     straightforward

     statement

    .

     The

     key

     details

     provided

     are

     the

     capital

     city

     and

     the

     country

     it

     belongs

     to

    ,

     making

     it

     clear

     and

     relevant

     to

     the

     query

    .

     The

     response

     is

     clear

    ,

     un

    ambiguous

    ,

     and

     does

     not

     contain

     any

     errors

     or

     missing

     information

    .

     The

     statement

     ful

    fills

     the

     requirement

     of

     providing

     a

     factual

     statement

     about

     Paris

     and

     its

     location

     within

     France

    .

     Therefore

    ,

     it

     is

     the

     most

     appropriate

     and

     accurate

     response

     to

     the

     given

     query

    .

     
    


    The

     provided

     answer

     could

     be

     rewritten

     as

    :
    


    "The

     capital

     of

     France

     is

     Paris

    ."
    


    This

     reform

    ulation

     maintains

     the

     original

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     unpredictable

     and

     uncertain

    ,

     but

     here

     are

     some

     possible

     trends

     that

     could

     occur

    :
    


    1

    .

     Increased

     AI

     for

     everyday

     tasks

    :

     AI

     is

     becoming

     more

     sophisticated

    ,

     allowing

     for

     more

     personalized

     and

     efficient

     solutions

     in

     areas

     like

     customer

     service

    ,

     healthcare

    ,

     and

     retail

    .
    


    2

    .

     AI

     for

     autonomous

     vehicles

    :

     Self

    -driving

     cars

    ,

     which

     are

     increasingly

     available

    ,

     could

     replace

     humans

     in

     the

     car

     industry

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     could

     be

     used

     to

     develop

     more

     accurate

     diagnosis

     and

     treatment

     plans

    ,

     as

     well

     as

     to

     predict

     which

     patients

     are

     at

     risk

     for

     certain

     diseases

    .
    


    4

    .

     AI

     for

     environmental

     monitoring

    :

     AI

     could

     be

     used

     to

     monitor

     pollution

     levels

    ,

     track

     def

    orestation

    ,

     and

     predict

     future

     climate

     changes

    



```python
llm.shutdown()
```
