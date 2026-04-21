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
    [2026-04-21 21:56:11] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.68it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.67it/s]


    2026-04-21 21:56:16,073 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 21:56:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:26,  2.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:26,  2.58s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:26,  2.58s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:26,  2.58s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 14.31it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 14.31it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 14.31it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 14.31it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 14.31it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:02<00:02, 14.31it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:02<00:02, 14.31it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:02<00:02, 14.31it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:02<00:02, 14.31it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:02<00:02, 14.31it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 22.76it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 27.22it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 27.22it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 27.22it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 27.22it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 27.22it/s]

    Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 27.22it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 27.22it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 27.22it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 27.22it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:03<00:00, 27.22it/s]Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:03<00:00, 27.22it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:03<00:00, 38.22it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 50.93it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 50.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.23 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.20 GB):   3%|▎         | 2/58 [00:00<00:04, 13.78it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:04, 13.78it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:04, 13.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.19 GB):   7%|▋         | 4/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.19 GB):   7%|▋         | 4/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.19 GB):   7%|▋         | 4/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.19 GB):  10%|█         | 6/58 [00:00<00:03, 15.91it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):  10%|█         | 6/58 [00:00<00:03, 15.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.19 GB):  10%|█         | 6/58 [00:00<00:03, 15.91it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=74.18 GB):  10%|█         | 6/58 [00:00<00:03, 15.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.32it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  21%|██        | 12/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  21%|██        | 12/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  21%|██        | 12/58 [00:00<00:02, 21.91it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  21%|██        | 12/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  21%|██        | 12/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.13 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.03it/s]Capturing num tokens (num_tokens=960 avail_mem=74.14 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.03it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.14 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.03it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.03it/s]Capturing num tokens (num_tokens=768 avail_mem=74.09 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.03it/s]Capturing num tokens (num_tokens=704 avail_mem=74.09 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.03it/s]Capturing num tokens (num_tokens=704 avail_mem=74.09 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.92it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.92it/s]Capturing num tokens (num_tokens=576 avail_mem=74.08 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.92it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.92it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.92it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.92it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.92it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.13it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.13it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.13it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.13it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.13it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.13it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.13it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=224 avail_mem=74.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.62it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=112 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.09it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.53it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.53it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.53it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.53it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.53it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.01 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.53it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.53it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  97%|█████████▋| 56/58 [00:01<00:00, 48.73it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  97%|█████████▋| 56/58 [00:01<00:00, 48.73it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  97%|█████████▋| 56/58 [00:01<00:00, 48.73it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 36.29it/s]


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
    Generated text:  Samantha and I’m from New York, the United States. I am currently a Ph.D. candidate at the University of Georgia and am interested in using artificial intelligence to improve quality of life. My work focuses on developing algorithms to make data-driven decisions, and I also study how computers, especially artificial intelligence, can be used to improve the quality of life.
    My research explores how artificial intelligence can be used to improve quality of life, and I work on developing methods to analyze a large number of real-world data sets, such as population data, to improve the quality of life for patients. I also work on developing algorithms to make decisions in healthcare
    ===============================
    Prompt: The president of the United States is
    Generated text:  21 years younger than his president-elect. If the president of the United States now is 72 years old, how old will the president of the United States be when his president-elect is born?
    To determine the age of the president of the United States when his president-elect is born, we can follow these steps:
    
    1. Identify the current age of the president of the United States.
    2. Determine the age of the president of the United States's president-elect.
    3. Calculate the president of the United States's age when his president-elect is born.
    
    From the problem, we know that the president of the United States is
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. Nice
    C. London
    D. Berlin
    
    To determine the capital of France, we need to recall the information about its location and known facts. The capital of France is Paris.
    
    Here is the reasoning step by step:
    
    1. Identify the capital of France. The capital of France is Paris.
    2. Confirm the capital's location. Paris is located in the South-Eastern region of France.
    3. Confirm the capital's country. France is a country, but it is typically referred to as "France" rather than "France" itself.
    
    Based on the above reasoning, the correct answer
    ===============================
    Prompt: The future of AI is
    Generated text:  good, but it is also uncertain
    
    The future of AI is good, but it is also uncertain
    
    By: Alessandro Pugliese
    Published: December 15, 2020
    
    The future of AI is good, but it is also uncertain
    
    As the world becomes more interconnected, the trade-off between speed and quality in AI is becoming more pronounced. In particular, it is becoming difficult to achieve a more reliable and accurate solution, which is essential to develop and build robust algorithms that are able to serve their intended purpose. This is a pressing issue, since the world as we know it is changing rapidly.
    
    


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. I enjoy [insert a short, positive description of your hobbies or interests]. What's your favorite hobby or activity? I love [insert a short, positive description of your favorite activity]. What's your favorite book or movie? I love [insert a short, positive description of your favorite book or movie]. What's your favorite place to go? I love [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a bustling metropolis with a rich history and culture, and it is a popular tourist destination. The city is known for its fashion, art, and cuisine, and it is a major economic hub for France. Paris is a city that is a must-visit for anyone interested in French culture and history. It is also a city that is known for its food, with its famous croissants
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical AI: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy. AI developers will need to prioritize these concerns and ensure that their systems are designed to be fair and transparent.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs
    


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
    Generated text:  [insert your name]. I’m a [insert your profession or occupation] who has been living in the [insert location] for the past [insert number of years] years, and I’ve always been passionate about [insert one or two activities or hobbies you enjoy doing]. I believe in [insert one or two principles or values that guide my actions and decisions], and I believe that [insert a single statement that reflects your beliefs or values]. I believe that my life is [insert one or two words that describe what I try to achieve or accomplish]. As a [insert your occupation or profession], I’m here to [insert one or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the largest and most populous city in the country.
    
    Paris is a historic and cultural center, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a vibrant and thriving city, with a thriving food and fashion industry, and a rich history dating back to the Roman Empire. Paris is often referred to as the "City of Light" due to its historical significance and the many artistic and cultural events that take place there. The city is also known for its contributions to fashion, including the famous couture couture and the Paris fashion week. Overall,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a combination of ongoing developments in technology, changing human behavior, and evolving societal needs. Here are some possible trends to consider:
    
    1. AI will become more ubiquitous: As the use of AI becomes more widespread, it's likely that we'll see more widespread adoption of AI in areas like healthcare, transportation, and consumer electronics. This could lead to a more integrated AI ecosystem where AI is integrated with other technologies like robotics, sensors, and data analytics.
    
    2. AI will become more ethical: As more people become aware of the risks associated with AI, there will be a greater push to develop ethical guidelines for AI.


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

     [

    occupation

    ]

     with

     [

    number

    ]

     years

     of

     experience

    .

     I

    'm

     a

     professional

     who

     is

     passionate

     about

     [

    what

     you

     like

     to

     do

    ].

     
    


    I

     enjoy

     [

    why

     you

     like

     it

    ]

     and

     have

     always

     wanted

     to

     [

    what

     you

     want

     to

     achieve

    ].

     I

    'm

     also

     [

    why

     you

     do

     it

    ]

     and

     my

     ultimate

     goal

     is

     to

     [

    what

     you

     want

     to

     achieve

    ].

     
    


    And

     while

     my

     work

     may

     seem

     to

     me

     to

     be

     a

     job

    ,

     I

     see

     it

     as

     a

     career

     and

     it

    's

     the

     love

     of

     my

     life

    .

     I

    'm

     ready

     to

     take

     on

     any

     challenge

    ,

     no

     matter

     how

     big

     or

     small

    ,

     because

     I

     know

     that

     I

    'm

     made

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     one

     of

     the

     largest

     in

     Europe

    .

     The

     city

     is

     located

     on

     the

     Se

    ine

     River

    ,

     and

     is

     home

     to

     the

     ancient

     city

     of

     Paris

    ,

     as

     well

     as

     many

     modern

     landmarks

     such

     as

     Notre

     Dame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     known

     for

     its

     rich

     cultural

     heritage

    ,

     including

     the

     presence

     of

     several

     world

    -ren

    owned

     museums

     and

     galleries

    ,

     including

     the

     Lou

    vre

    ,

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     Mus

    ée

     d

    '

    Art

     Moder

    ne

    .

     Additionally

    ,

     Paris

     is

     known

     for

     its

     iconic

     fashion

     industry

    ,

     with

     many

     high

    -end

     bout

    iques

     and

     design

     houses

     located

     in

     the

     city

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     highly

     dynamic

     and

     unpredictable

     one

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     in

     various

     industries

    :

     AI

     is

     becoming

     more

     integrated

     into

     various

     industries

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     manufacturing

    .

     This

     trend

     is

     likely

     to

     continue

     in

     the

     coming

     years

     as

     AI

     continues

     to

     be

     integrated

     into

     more

     areas

     of

     work

    .
    


    2

    .

     AI

     becomes

     more

     personalized

    :

     AI

     is

     becoming

     more

     personalized

     as

     it

     learns

     to

     understand

     and

     adapt

     to

     individual

     needs

     and

     preferences

    .

     As

     a

     result

    ,

     we

     may

     see

     AI

     systems

     that

     are

     more

     personalized

     to

     meet

     the

     specific

     needs

     of

     individual

     users

    .
    


    3

    .

     AI

     becomes

     more

     integrated

     with

     the

     environment

    :

    



```python
llm.shutdown()
```
