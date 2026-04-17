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
    [2026-04-17 16:10:45] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.31it/s]


    2026-04-17 16:10:49,998 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 16:10:49] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:30,  1.78it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.78it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.57it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.57it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.57it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.57it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.57it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.57it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.57it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.57it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.57it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 19.71it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 19.71it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 19.71it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 19.71it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 19.71it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 19.71it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 19.71it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 19.71it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 19.71it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:03<00:01, 19.71it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 28.86it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 28.86it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 28.86it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 28.86it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 28.86it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 28.86it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 28.86it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 28.86it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 28.86it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 36.84it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 36.84it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 36.84it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 36.84it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 36.84it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 36.84it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 36.84it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:03<00:00, 36.84it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:03<00:00, 36.84it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 44.62it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 44.62it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 44.62it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 44.62it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 44.62it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 44.62it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 44.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.75it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.75it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=118.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.39 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.39 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.37 GB):  31%|███       | 18/58 [00:00<00:01, 35.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.37 GB):  31%|███       | 18/58 [00:00<00:01, 35.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.35 GB):  31%|███       | 18/58 [00:00<00:01, 35.16it/s]

    Capturing num tokens (num_tokens=960 avail_mem=118.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.16it/s] Capturing num tokens (num_tokens=896 avail_mem=118.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.16it/s]Capturing num tokens (num_tokens=896 avail_mem=118.36 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=832 avail_mem=118.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=768 avail_mem=118.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=704 avail_mem=118.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=640 avail_mem=118.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=576 avail_mem=118.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.76it/s]Capturing num tokens (num_tokens=576 avail_mem=118.34 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.51it/s]Capturing num tokens (num_tokens=512 avail_mem=118.33 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.51it/s]Capturing num tokens (num_tokens=480 avail_mem=118.35 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.51it/s]

    Capturing num tokens (num_tokens=448 avail_mem=118.35 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.51it/s]Capturing num tokens (num_tokens=416 avail_mem=118.34 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.51it/s]Capturing num tokens (num_tokens=384 avail_mem=118.34 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.51it/s]Capturing num tokens (num_tokens=384 avail_mem=118.34 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.86it/s]Capturing num tokens (num_tokens=352 avail_mem=118.34 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.86it/s]Capturing num tokens (num_tokens=320 avail_mem=118.33 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.86it/s]Capturing num tokens (num_tokens=288 avail_mem=118.33 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.86it/s]Capturing num tokens (num_tokens=256 avail_mem=118.33 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.86it/s]Capturing num tokens (num_tokens=240 avail_mem=118.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=240 avail_mem=118.32 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=224 avail_mem=118.32 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.58it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.32 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=192 avail_mem=118.32 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=176 avail_mem=118.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=160 avail_mem=118.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.58it/s]Capturing num tokens (num_tokens=160 avail_mem=118.31 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.28it/s]Capturing num tokens (num_tokens=144 avail_mem=118.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.28it/s]Capturing num tokens (num_tokens=128 avail_mem=118.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.28it/s]Capturing num tokens (num_tokens=112 avail_mem=118.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.28it/s]Capturing num tokens (num_tokens=96 avail_mem=118.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.28it/s] Capturing num tokens (num_tokens=80 avail_mem=118.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.28it/s]

    Capturing num tokens (num_tokens=80 avail_mem=118.29 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=64 avail_mem=118.29 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=48 avail_mem=118.29 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=32 avail_mem=118.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=28 avail_mem=118.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=24 avail_mem=118.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=24 avail_mem=118.27 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.65it/s]Capturing num tokens (num_tokens=20 avail_mem=118.27 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.65it/s]Capturing num tokens (num_tokens=16 avail_mem=118.27 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.65it/s]Capturing num tokens (num_tokens=12 avail_mem=117.00 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.65it/s]Capturing num tokens (num_tokens=8 avail_mem=116.90 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.65it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=116.90 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.65it/s]Capturing num tokens (num_tokens=4 avail_mem=116.90 GB): 100%|██████████| 58/58 [00:01<00:00, 43.12it/s]Capturing num tokens (num_tokens=4 avail_mem=116.90 GB): 100%|██████████| 58/58 [00:01<00:00, 38.58it/s]


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
    Generated text:  Maria, and I'm an experienced author who specializes in educational materials. As a passionate educator, I am committed to ensuring that every student in my classroom has the opportunity to learn. I believe that education should be accessible and inclusive, and I am always looking for ways to improve my teaching methods and materials to better serve my students. In addition, I am a fan of incorporating technology and digital tools to enhance learning and engage students in a deeper way.
    My ultimate goal is to create materials that not only teach students about complex subjects but also to inspire them to think critically, question, and solve problems. I am passionate about using a variety of
    ===============================
    Prompt: The president of the United States is
    Generated text:  traveling on a plane that is divided into two sections: one section is for the executive team, and the other section is for the non- executive team. The executive team takes up 1/5 of the seats on the plane, and the non- executive team takes up 1/3 of the seats. If there are 120 seats on the plane, how many seats are left for the non- executive team?
    
    To determine how many seats are left for the non- executive team, we need to calculate the number of seats taken by the executive team and the non- executive team, and then subtract these from the total
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris. It is the 9th largest city in the world. Paris was founded by the ancient Gauls and was made a city in 733 AD by the King of France Clovis I. A city council was established there in 842 AD. Paris was an important trading city until the Middle Ages. It had a high number of important people who lived there. Most of them were Christians who lived there for centuries. The French Revolution took place in Paris in the 1780s. Paris was bombed by the Germans in 1940. The French Resistance fought against the Nazis
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but not for everyone. Here's what you should know.
    BY KASSRA NABESH NOOR, SITHA LAXI
    In recent months, the AI community has been inundated with a flood of information about the potential of AI. These are a mixture of the positive and the negative. The good news is that the future of AI is bright and that AI is here to change the world. The bad news is that, despite its promise, the world is not ready for AI yet. This article will take a look at some of the misconceptions about AI, and explain how AI is here to change the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character, such as "funny, adventurous, or curious"] who is [insert a short description of your character, such as "a tech-savvy, data-driven individual" or "a passionate, creative, and independent person"]. I enjoy [insert a short description of your character's hobbies or interests, such as "reading, cooking, or playing music"]. I'm always looking for new experiences and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. 
    
    (Note: The statement provided is a factual statement about Paris, not a fictional one. The Eiffel Tower is a famous landmark in Paris, and Notre-Dame Cathedral is a significant religious site in the city.) 
    
    The statement is concise and factual, providing a clear and accurate description of Paris's capital city. However, it could be expanded to include more details about Paris's cultural significance, such as its role in French history, its influence on global culture, or its role in the French language. For example:
    
    "The capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI, there will be a greater emphasis on ethical AI. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased investment in research and development to address ethical concerns.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, from smartphones and computers to healthcare and transportation. As more of these technologies become integrated, there
    


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
    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and discuss how I can help you achieve your goals. Let's talk about [reason for your introduction].
    Hello, my name is [Name], and I'm a [job title] at [company name]. I'm excited to meet you and discuss how I can help you achieve your goals. Let's talk about [reason for your introduction]. I'm a [introduction, e.g., "I'm a [job title] at [company name]"] who specializes in [field of expertise or services]. I believe in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the North West of the country. It is a historic city with a rich history and a vibrant cultural scene. The city is known for its beautiful art and architecture, as well as its delicious cuisine and lively nightlife. It is the largest city in France by population and is home to many important museums, theaters, and other cultural institutions. Paris is a major hub for international business and finance, and is the birthplace of the French Revolution and the foundation of French democracy. It is also known for its stunning views of the surrounding countryside and its unique blend of classical and modern architecture. The city is home to numerous universities and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undoubtedly going to be influenced by a wide range of technological advancements and changes in society. Here are some possible future trends in artificial intelligence:
    
    1. Increased accuracy and precision in AI applications: With advancements in machine learning, AI systems will become more accurate and precise in their decision-making. This will allow for a wider range of applications, from healthcare to financial services to customer support.
    
    2. Integration of AI into daily life: As AI becomes more integrated into our daily lives, we will see more automation and smarter devices. This will lead to a more efficient and convenient way of using technology, such as in transportation, home automation, and healthcare.
    
    


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

     first

     name

    ],

     and

     I

    'm

     a

     [

    insert

     profession

     or

     occupation

    ]

     with

     a

     passion

     for

     [

    insert

     a

     hobby

     or

     interest

     of

     your

     choice

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     As

     a

     [

    insert

     a

     profession

     or

     occupation

    ],

     I

     love

     [

    insert

     something

     that

     sparks

     my

     interest

    ,

     such

     as

     reading

    ,

     music

    ,

     or

     cooking

    ].

     I

     believe

     in

     [

    insert

     a

     value

     or

     principle

     that

     guides

     my

     life

    ,

     such

     as

     [

    insert

     a

     value

     or

     principle

    ]).

     I

     enjoy

     [

    insert

     something

     that

     I

     hope

     others

     can

     learn

     from

    ,

     such

     as

     [

    insert

     a

     skill

     or

     habit

     that

     I

     think

     makes

     me

     unique

     or

     special

    ,

     such

     as

     [

    insert

     a

     skill

     or

     habit

     that

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     factual

     statement

     is

    :

     Paris

     is

     the

     capital

     of

     France

    .

     
    


    I

     apologize

    ,

     but

     there

     is

     a

     logical

     error

     in

     my

     previous

     statement

    .

     As

     of

     

    2

    0

    2

    3

    ,

     Paris

     is

     the

     capital

     of

     France

     but

     has

     not

     been

     called

     such

     for

     centuries

    .

     The

     capital

     city

     has

     a

     rich

     and

     diverse

     history

    ,

     which

     has

     been

     passed

     down

     through

     generations

    .

     However

    ,

     the

     statement

     I

     initially

     provided

     was

     accurate

     up

     until

     the

     

    1

    9

    th

     century

    .

     Paris

     was

     indeed

     the

     capital

     of

     France

     from

     

    1

    8

    0

    4

     until

     

    1

    9

    6

    9

    .

     It

     was

     then

     renamed to

     Paris

     after

     the

     French

     revolution

     of

     

    1

    7

    8

    9

    ,

     which

     led

     to

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     with

     many

     possibilities

     ahead

     for

     the

     technology

    .

     Here

     are

     some

     possible

     trends

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     As

     AI

     learns

     to

     understand

     human

     emotions

     and

     preferences

    ,

     it

     will

     be

     able

     to

     provide

     more

     tailored

     experiences

     and

     recommendations

    ,

     leading

     to

     a

     more

     personalized

     user

     experience

    .
    


    2

    .

     Machine

     Learning

     and

     Deep

     Learning

    :

     These

     advanced

     AI

     techniques

     will

     continue

     to

     evolve

    ,

     leading

     to

     even

     greater

     capabilities

     and

     accuracy

     in

     solving

     complex

     problems

    .
    


    3

    .

     AI

     for

     Healthcare

    :

     With

     the

     increasing

     amount

     of

     personal

     health

     data

     being

     collected

    ,

     AI

     will

     be

     used

     to

     develop

     better

     medical

     treatments

     and

     diagnostic

     tools

    .
    


    4

    .

     Autonomous

     Vehicles

    :

     As

     AI

     improves

    ,

     autonomous

     vehicles

     will

     become

     more

     common

    ,

     leading

    



```python
llm.shutdown()
```
