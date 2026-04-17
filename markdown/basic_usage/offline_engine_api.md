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
    [2026-04-17 20:39:34] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.02it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.00it/s]


    2026-04-17 20:39:39,382 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 20:39:39] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.71it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 13.00it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 13.00it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 13.00it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 13.00it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 13.00it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 13.00it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 13.00it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 13.00it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 13.00it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.23it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.23it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.23it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.23it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.23it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.23it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.23it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.23it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.23it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:03<00:01, 20.23it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 29.42it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 29.42it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 29.42it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 29.42it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 29.42it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 29.42it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 29.42it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 29.42it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 29.42it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:03<00:00, 29.42it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 38.49it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 38.49it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 38.49it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 38.49it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 38.49it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 38.49it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 38.49it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 38.49it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 38.49it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:03<00:00, 38.49it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 47.56it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 47.56it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 47.56it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 47.56it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 47.56it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.70it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.70it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.73 GB):   7%|▋         | 4/58 [00:00<00:03, 15.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   7%|▋         | 4/58 [00:00<00:03, 15.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.72 GB):   7%|▋         | 4/58 [00:00<00:03, 15.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.72 GB):   7%|▋         | 4/58 [00:00<00:03, 15.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.72 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.72 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.72 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.17it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.05it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.05it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.05it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.05it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.05it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.61it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.61it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.61it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.61it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.97it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.97it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.97it/s]

    Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.97it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.97it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.97it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  50%|█████     | 29/58 [00:00<00:00, 37.91it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  50%|█████     | 29/58 [00:00<00:00, 37.91it/s]Capturing num tokens (num_tokens=448 avail_mem=120.24 GB):  50%|█████     | 29/58 [00:00<00:00, 37.91it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:00<00:00, 37.91it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:00<00:00, 37.91it/s]Capturing num tokens (num_tokens=352 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:01<00:00, 37.91it/s]Capturing num tokens (num_tokens=352 avail_mem=120.23 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.49it/s]

    Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.50it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.50it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.50it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.50it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.50it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  67%|██████▋   | 39/58 [00:01<00:00, 40.50it/s]

    Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.40it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.40it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.40it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.40it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.40it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.40it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=48 avail_mem=120.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.41it/s]

    Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=12 avail_mem=120.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=8 avail_mem=120.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.77it/s] Capturing num tokens (num_tokens=4 avail_mem=120.12 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=4 avail_mem=120.12 GB): 100%|██████████| 58/58 [00:01<00:00, 34.04it/s]


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
    Generated text:  Sophia and I'm a 25 year old woman who is a therapist, and I have an interest in kung fu. I've always been fascinated with the martial arts and how it's a form of therapy and how it has the potential to promote physical and mental health.
    
    I'm curious about the differences between kung fu and yoga. Could you provide me with some information on this topic?
    Yes, definitely! Kung fu and yoga are both types of martial arts that have been practiced in various cultures for thousands of years. While they both involve physical exercise and have their own unique styles, they also have some similarities and differences.
    
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected by voters in a ______.
    electoral district
    
    The president of the United States is elected by voters in an electoral district. The President of the United States is directly elected by the citizens of the United States. The election of the president happens every four years. The term of office of the president is four years. The president has to win 2/3 majority in the legislature to avoid being removed by the legislature.
    
    What did the 1992 election result show? The 1992 election resulted in a significant increase in the number of votes for the Democratic Party compared to the Republican Party. This shift in the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the most visited city in the world. The city attracts millions of tourists every year. Paris is one of the world’s most expensive cities. But the city’s charm attracts many visitors. Paris has its own unique and fun activities. The city also has some strange things too. Paris, France is a city where tourists often feel like they are a ghost. The city is very old. It has a long history. Many people believe that Paris has its own fairy tale. The city is a little bit different from other cities. There are many cafes and restaurants in Paris. The atmosphere is lively. Many people go there after
    ===============================
    Prompt: The future of AI is
    Generated text:  here and it's changing the way we live, work, and play. With the increasing availability of AI, there's a lot to learn and explore. In this article, we'll discuss how AI is changing the way we interact with technology, as well as its impact on the future of work. We'll also take a look at the various types of AI that are currently being developed and the opportunities and challenges they present. So, if you're a tech enthusiast or a business professional who's looking to stay ahead of the curve, you need to read this article to see how AI is transforming the world around us. Let's dive in


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I have a [job title] at [company name]. I'm [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I have a [job title] at [company name]. I'm [job title
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in the country and the seat of government, administration, and culture. The city is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a popular tourist destination, attracting millions of visitors each year. Paris is a cultural hub and a major economic center in France, and its influence extends beyond its borders. The city is home to many international organizations and institutions, including UNESCO and the French Academy of Sciences. Paris is a city of contrasts, with its modern
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to advance, we can expect to see even more sophisticated applications in healthcare, such as personalized medicine, drug discovery, and disease diagnosis.
    
    2. AI in manufacturing: AI is already being used in manufacturing to improve efficiency, reduce costs, and increase productivity. As AI technology continues to advance, we can expect to see even more sophisticated
    


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
    Generated text:  [Name]. I'm a [Occupation] from [Location], with [Number] years of experience in this field. I'm always looking to learn something new and stay up-to-date with the latest trends and technologies. I'm a [Number] time management expert and always aim to prioritize my tasks and stay organized. I'm also a [Number] leader, and I'm always willing to lend a hand when needed. I'm friendly, energetic, and always ready to help. I'm confident in my abilities and eager to make a difference in the world. Thank you for asking, and I hope to meet you in person!
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Please note: This answer is general and should not constitute an endorsement of any political party, organization or individual. The content of this answer is not subject to change at any time without notice. This answer is not to be used, transmitted, copied or distributed, in any form or by any means without express written permission from the author. 
    Remember to include your answers in a comment section below. 
    
    #Paris #ParisFacts
    Paris, officially known as the Île-de-France metropolitan area, is the capital of France. The city is located on the western bank of the Seine River, and is the third
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see a number of significant trends emerge, including:
    
    1. Increased Automation: AI is already becoming more capable of performing tasks in a wide range of fields, from production lines to customer service. As technology continues to advance, we can expect automation to become even more prevalent, as AI is able to perform tasks more efficiently and accurately than human workers.
    
    2. Emphasis on Explainability: As AI systems become more sophisticated, we will see an increased focus on how the system makes decisions and what factors led to those decisions. This will encourage developers to create systems that can be more transparent and explainable, making it easier for humans to


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

     Jane

    ,

     and

     I

    'm

     a

     professional

     book

    keeper

     at

     the

     prestigious

     library

    .

     I

    've

     been

     with

     the

     company

     for

     four

     years

    ,

     and

     I

     enjoy

     helping

     patrons

     find

     the

     information

     they

     need

    .
    


    That

    's

     great

     to

     hear

    !

     What

     do

     you

     do

     at

     the

     library

    ?
    


    As

     a

     book

    keeper

    ,

     I

     work

     with

     the

     staff

     to

     ensure

     that

     the

     books

     and

     records

     are

     accurate

     and

     up

    -to

    -date

    .

     I

     also

     help

     with

     the

     inventory

     count

     and

     make

     sure

     that

     all

     patrons

     have

     the

     correct

     materials

     and

     information

    .

     I

     enjoy

     helping

     patrons

     find

     the

     information

     they

     need

    ,

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     improve

     the

     library

    's

     operations

    .

     What

     other

     things

     do

     you

     do

     at

     the

     library

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     capital

     of

     the

     country

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     and

     has

     a

     rich

     history

    ,

     including

     a

     tumult

    uous

     past

     of

     its

     many

     em

    pires

     and

     conflicts

    .

     The

     city

     is

     known

     for

     its

     architecture

    ,

     cuisine

    ,

     and

     cultural

     attractions

    ,

     including

     the

     Lou

    vre

     Museum

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     an

     iconic

     landmark

     that

     has

     been

     a

     symbol

     of

     the

     city

     since

     its

     construction

     in

     

    1

    8

    8

    9

    .

     The

     city

     is

     also

     famous

     for

     its

     fashion

     industry

    ,

     which

     has

     played

     an

     important

     role

     in

     the

     country

    's

     cultural

     and

     economic

     development

    .

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     full

     of

     possibilities

    ,

     as

     it

     continues

     to

     evolve

     and

     develop

     at

     an

     unprecedented

     rate

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

     and

     deep

     learning

    :

     With

     the

     help

     of

     big

     data

     and

     advanced

     algorithms

    ,

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     understand

     and

     interpret

     complex

     data

    .

     Machine

     learning

     models

     will

     become

     more

     sophisticated

    ,

     with

     algorithms

     that

     can

     learn

     from

     examples

     and

     improve

     their

     performance

     over

     time

    .
    


    2

    .

     Increased

     integration

     with

     human

     experts

    :

     AI

     will

     continue

     to

     integrate

     more

     deeply

     with

     human

     experts

    ,

     allowing

     them

     to

     provide

     insights

     and

     guidance

     to

     the

     AI

     system

    .

     This

     could

     lead

     to

     more

     effective

     and

     sustainable

     AI

     systems

    .
    


    3

    .

    



```python
llm.shutdown()
```
