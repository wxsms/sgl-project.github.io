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
    [2026-04-22 21:30:22] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.67it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.67it/s]


    2026-04-22 21:30:28,069 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 21:30:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:57,  3.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:57,  3.11s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<02:57,  3.11s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<02:57,  3.11s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.62it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.62it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:33,  1.62it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:33,  1.62it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:33,  1.62it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:33,  1.62it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:33,  1.62it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:09,  5.05it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:09,  5.05it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:09,  5.05it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:09,  5.05it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:09,  5.05it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:09,  5.05it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:09,  5.05it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:09,  5.05it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:09,  5.05it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:09,  5.05it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:03<00:09,  5.05it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:03, 12.28it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:03, 12.28it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:03, 12.28it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:03, 12.28it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:03, 12.28it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:03, 12.28it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:03, 12.28it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:03, 12.28it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:03, 12.28it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:03, 12.28it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 19.88it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 19.88it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 19.88it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 19.88it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 19.88it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 19.88it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 19.88it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 19.88it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 19.88it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 19.88it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 28.29it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 28.29it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 38.56it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 38.56it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 38.56it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 38.56it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 38.56it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 38.56it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 38.56it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 38.56it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 38.56it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 38.56it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 38.56it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.01 GB):   3%|▎         | 2/58 [00:00<00:02, 19.25it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.01 GB):   3%|▎         | 2/58 [00:00<00:02, 19.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.01 GB):   3%|▎         | 2/58 [00:00<00:02, 19.25it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):   9%|▊         | 5/58 [00:00<00:02, 22.31it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  21%|██        | 12/58 [00:00<00:01, 24.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  21%|██        | 12/58 [00:00<00:01, 24.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  21%|██        | 12/58 [00:00<00:01, 24.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 24.84it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.71 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.72it/s] Capturing num tokens (num_tokens=896 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.72it/s]

    Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.15it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.15it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.15it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.15it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.15it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 33.15it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.54it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.54it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.54it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.34it/s]

    Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.08it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.96it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.96it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.96it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.96it/s]

    Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.96it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.96it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.87it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 35.06it/s]


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
    Generated text:  Aarti and I am a medical student. I have recently started my second year. My first year was very tough as I had a relative who was very ill and we had to stay with them. After the hospitalisation was over, I had a week of medical training which went well. However, I had to face the circumstances of my second year as it was my first year and I was a 15-year-old student. I had to pick up my first year's lectures and exams as my first year was over. This was challenging. It was really hard for me to manage my time and everything else, including the exams.
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to persuade the U.S. Congress to pass a bill that would make it easier to stay healthy.
    Why would the president want to make it easier to stay healthy?
    A) Because he is getting sick too
    B) Because he wants to get more exercise
    C) Because he wants to go on vacation
    D) Because he wants to go to work
    Answer the following question: What does the President want to do with the bill? To answer this question, you should look at the information provided in the context. The relevant information is: The president of the United States is trying to persuade the U. S. Congress to pass
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. (判断题)
    
    错误
    
    下列关于“资本公积”科目的说法中，不正确的有（ ）。
    
    资本公积包括资本溢价、股本溢价、资本溢价、股本溢价等
    
    在资本公积中，属于资本溢价的是（）。
    
    股本溢价
    
    账户的贷方用来记录（ ）。
    
    增加的金额
    
    下面关于股票投资的表述错误的是（ ）。
    
    普通股票的收益通常比优先股票大
    
    关于“资产=负债+所有者权益”这一会计恒等式，以下说法正确的是（ ）。
    
    资产和负债需要同时记录，所有者权益在
    ===============================
    Prompt: The future of AI is
    Generated text:  not in the cloud
    Technology and innovation are advancing at an unprecedented rate, and as a result, AI is becoming more integrated in our daily lives. This has led to the emergence of AI technologies such as machine learning, deep learning, and natural language processing. The technologies that have emerged in recent years have achieved remarkable success and have made a positive impact on our daily lives.
    In the context of the future of AI, it is important to recognize that the technology is evolving faster than ever before and that it is not just a matter of where the technology is stored, but also the way it is used. The future of AI is not in


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? I'm a [age] year old, [gender] and I have a [number] degree in [field of study]. I'm a [occupation] who has always been passionate about [interest or hobby]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [mention a hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic center that plays a significant role in French politics and society. The city is also known for its rich history, including the Roman and medieval periods, and its role in the French Revolution and the French Revolution. Paris is a popular tourist destination, with millions of visitors annually. The city is also known for its cuisine, including French cuisine, and its fashion industry. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability. AI developers will need to be more mindful of the potential impact of their technology on society and work to ensure that it is used in a responsible and ethical manner.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced decision-making
    


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
    Generated text:  [Name], and I am a [job title] at [company name]. I am currently working in [current job role] at [company name]. In my free time, I enjoy [things that bring me joy and fulfillment]. How can I be reached?
    Hello, my name is [Name], and I am a [job title] at [company name]. I am currently working in [current job role] at [company name]. In my free time, I enjoy [things that bring me joy and fulfillment]. How can I be reached?
    I'm [Name] and I'm excited to meet you. Feel free to reach
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris, the French capital, is a city located on the Île de la Cité, a crescent-shaped island in the Seine River. It is one of the world's most famous and oldest cities, with an extensive and diverse history dating back to the 7th century. Paris is known for its magnificent architecture, cultural richness, and historical landmarks. Its renowned fashion industry, including the iconic Eiffel Tower, and its romantic romanticism have made it a global destination. The city is home to numerous museums, theaters, and a vibrant arts scene, attracting millions of visitors each year. As the cultural and economic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid advancements in the areas of machine learning, natural language processing, robotics, autonomous vehicles, and the development of more advanced versions of existing AI technologies. In addition, there is likely to be an increase in the use of AI in industries such as healthcare, finance, and energy. As AI becomes more integrated into our daily lives, we can expect to see more widespread adoption of AI technologies in various sectors. In short, the future of AI looks bright and likely to play a critical role in shaping our world in the years ahead. 
    
    One of the most significant trends in the future of AI is the growing importance of


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

    position

    ]

     at

     [

    company

    ].

     I

     recently

     graduated

     from

     [

    un

    iversity

    ]

     and

     have

     been

     working

     hard

     to

     [

    what

     I

     hope

     to

     achieve

    ].

     I

     am

     passionate

     about

     [

    reason

     for

     passion

    ],

     and

     I

     believe

     that

     my

     skills

     and

     experience

     can

     contribute

     to

     [

    specific

     company

    's

     mission

    ].

     I

     am

     confident

     that

     I

     am

     a

     good

     fit

     for

     [

    company

    's

     position

    ]

     and

     look

     forward

     to

     contributing

     to

     their

     mission

    .


    Sure

    ,

     here

    's

     a

     short

    ,

     neutral

     self

    -int

    roduction

     for

     a

     fictional

     character

    :


    Hello

    ,

     my

     name

     is

     [

    Name

    ]

     and

     I

     am

     a

     [

    position

    ]

     at

     [

    company

    ].

     I

     recently

     graduated

     from

     [

    un

    iversity

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     an

     historic

     and

     cultural

     center

     with

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     era

    .

     The

     city

     is

     famous

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     Its

     cuisine

     is

     also

     renowned

     for

     its

     well

    -known

     dishes

     such

     as

     fo

    ie

     gras

    ,

     esc

    arg

    ot

    ,

     and

     cro

    iss

    ants

    .

     In

     terms

     of

     transport

    ,

     the

     city

     is

     connected

     by

     high

    -speed

     trains

     that

     run

     through

     its

     metro

     system

    .

     Paris

     is

     home

     to

     some

     of

     the

     world

    's

     most

     well

    -known

     museums

    ,

     including

     the

     Lou

    vre

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Mus

    ée

     des

     Arts

     Dé

    cor

    at

    ifs

    .

     The

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     quite

     dynamic

     and

     the

     possibilities

     for

     development

     are

     vast

    .

     Here

     are

     some

     potential

     trends

     we

     can

     expect

     to

     see

     in

     the

     field

     of

     artificial

     intelligence

    :
    


    1

    .

     Improved

     performance

     and

     efficiency

    :

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

     significant

     improvements

     in

     the

     performance

     and

     efficiency

     of

     AI

     systems

    .

     This

     could

     lead

     to

     more

     accurate

     predictions

    ,

     faster

     decision

    -making

    ,

     and

     improved

     user

     experience

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     already

     a

     reality

    ,

     with

     self

    -driving

     cars

     on

     the

     roads

     today

    .

     As

     the

     technology

     advances

    ,

     we

     can

     expect

     to

     see

     more

     autonomous

     vehicles

     on

     the

     roads

     in

     the

     future

    .

     This

     could

     have

     a

     huge

     impact

     on

     our

     transportation

     system

    ,

     as

     well

     as

    



```python
llm.shutdown()
```
