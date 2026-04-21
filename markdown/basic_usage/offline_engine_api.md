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
    [2026-04-21 21:00:43] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.51it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.51it/s]


    2026-04-21 21:00:47,014 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 21:00:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:29,  2.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:29,  2.63s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 14.03it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 14.03it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 14.03it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 14.03it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 14.03it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 14.03it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 14.03it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 14.03it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.16it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.16it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.16it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.16it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.16it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.16it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.16it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.16it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.16it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:03<00:01, 20.16it/s]Compiling num tokens (num_tokens=256):  47%|████▋     | 27/58 [00:03<00:01, 20.16it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 30.54it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 30.54it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 30.54it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 30.54it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 30.54it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 30.54it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 30.54it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 30.54it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 30.54it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:03<00:00, 30.54it/s]

    Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:03<00:00, 30.54it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:03<00:00, 41.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 14.93it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 14.93it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 15.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 15.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 15.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:03, 17.11it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:03, 17.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:03, 17.11it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:03, 17.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:02, 22.78it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.18it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.18it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.18it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.18it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.32it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.32it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.32it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.32it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.32it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.64it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.64it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.64it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.64it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.64it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.64it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=240 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=240 avail_mem=76.68 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.49it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.49it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.49it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.49it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.49it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.49it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.18it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=32 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.69it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.68it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 34.79it/s]


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
    Generated text:  Atsushi Matsumoto. I'm the founder of the Traveler's Tasty and I'm the author of "Traveler's Tasty: Eat Like a Chef."
    I'm from Tokyo, Japan. I have been studying Chinese and Japanese cuisines since I was 10 years old.
    I like to make dishes that are delicious and also elegant.
    I like to learn and try new foods.
    I have a passion for sharing experiences and trying new foods with my family and friends. I love to play a lot, especially golfing and tennis.
    I love to go out to eat in restaurants.
    I've been at a restaurant
    ===============================
    Prompt: The president of the United States is
    Generated text:  retiring and will be succeeded by the Vice President. How many different possible Vice Presidents would there be?
    There are 432 possible Vice Presidents that could be chosen by the Vice President.
    This is because there are 432 different years in which the Vice President could be elected by the President, and for each year, there are 432 different Vice Presidents that could be chosen.
    Therefore, the answer is \boxed{432}.The answer is: 432.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is a well-known city known for its beautiful architecture, world-class museums, and its unique ambiance. However, there is an intriguing fact that stands out from the rest, which has been the subject of much speculation and controversy. The capital of France has a unique tradition of holding the National Day, which is observed on September 14. Now, imagine a city where this tradition is not celebrated, but instead observed on September 16th. This city would have its own unique tradition of honoring the nation's founding day, which is celebrated on August 15th. 
    
    Given this situation, the city would not
    ===============================
    Prompt: The future of AI is
    Generated text:  in action, not just theories and research. In order to see this, we need to change the way we perceive, think, and learn about AI. The technology is advancing at an unprecedented pace, and the world is changing to adapt to the new world of AI. The future of AI is exciting and full of possibilities, and it is about to transform our way of life.
    In this article, we will explore the future of AI and how it is changing the way we live and work. We will discuss the latest developments in AI and how they are transforming our world. We will also discuss the challenges and ethical considerations related to AI,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I've been working here for [number of years] years. I'm a [job title] at [company name], and I've been working here for [number of years] years. I'm a [job title] at [company name], and I've been working here for [number of years] years. I'm a [job title] at [company name], and I've been working here for [number of years] years. I'm a [job title] at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of French Revolution and Napoleon Bonaparte, and its role in the French Revolution. It is also a popular tourist destination, attracting millions of visitors each year. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is home to many famous French artists, writers, and musicians, and its cuisine is renowned for its delicious
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do. This could lead to more efficient and effective decision-making, as well as more personalized and context-aware interactions with humans.
    
    2. Greater emphasis on ethical and social implications: As AI becomes more integrated with human intelligence, there will be increased pressure to consider the ethical and social implications of its use. This could lead to more stringent regulations and standards, as well as more transparent and accountable AI systems.
    
    
    


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
    Generated text:  [Your Name], and I'm an experienced professional with a unique perspective on [Industry/Field]. I've been working in [Field] for [Number of Years] years and have gained a wealth of knowledge and skills in [What You Have Gained Knowledge/Skills In]. I'm a hard worker, with a strong work ethic, and I'm passionate about [What You Are Passionate About]. I'm always looking for ways to improve myself and to stay up-to-date with new developments in the field. I'm confident in my abilities and I'm excited to share my knowledge and experience with others. Thank you for considering me for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, a vibrant and cosmopolitan city that is home to a wide variety of cultural institutions and events, including the annual Eiffel Tower Festival. The city is located in the heart of the French countryside and is known for its rich history, beautiful architecture, and diverse population. As one of the world’s most visited cities, Paris is a must-visit destination for anyone looking to explore France and its history. It has a rich cultural heritage, with art, music, and literature being some of the most important aspects of French culture. With its renowned museums, art galleries, and theaters, Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to see continued growth and development, driven by advances in computing power, increased availability of data, and advancements in machine learning and deep learning algorithms. Some potential future trends include:
    
    1. More advanced natural language processing: With the increasing availability of large datasets and the growth of neural networks, there is an increasing focus on building more advanced natural language processing systems that can understand and generate human-like language.
    
    2. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, but the potential for even more sophisticated AI systems in this area is expected to grow rapidly.
    
    3. Increased use of AI in transportation


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

    Your

     Name

    ],

     and

     I

    'm

     a

     [

    description

     of

     your

     character

    ,

     e

    .g

    .

     "

    a

     friendly

    ,

     helpful

    ,

     and

     curious

     person

     who

     enjoys

     exploring

     new

     places

     and

     learning

     about

     various

     cultures

    ."

    ]

     I

    'm

     here

     to

     make

     you

     feel

     welcome

     and

     help

     you

     feel

     more

     comfortable

    .

     If

     you

     have

     any

     questions

     or

     if

     you

     need

     assistance

    ,

     please

     don

    't

     hesitate

     to

     reach

     out

    .

     Remember

    ,

     we

    're

     in

     this

     together

    ,

     and

     I

    'm

     here

     to

     support

     you

     every

     step

     of

     the

     way

    .

     #

    Greetings

     #

    Be

    Well

    .

     #

    Friendly

    .

     #

    Support

    ive

    .

     #

    Welcome.

     #

    New

    .

     #

    C

    ultural

    .

     #

    Expl

    oring

    .

     #

    Expl

    oring

    .

     #

    Learn

    .

     #

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     located

     in

     the

     Î

    le

     de

     la

     C

    ité

     on

     the

     north

     bank

     of

     the

     Se

    ine

     River

    ,

     and

     is

     the

     most

     populous

     city

     in

     the

     country

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     food

    ,

     as

     well

     as

     its

     status

     as

     the

     world

    's

     most

     populous

     city

    .

     It

     is

     also

     a

     major

     economic

     and

     political

     center

    ,

     and

     hosts

     numerous

     attractions

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Lou

    vre

     Museum

    .

     Paris

     is

     the

     seat

     of

     government

     for

     the

     French

     Republic

     and

     is

     home

     to

     numerous

     museums

    ,

     art

     galleries

    ,

     and

     concert

     halls

    .

     Its

     nickname

     "

    The

     City

     of

     Light

    "

     refers

     to

     its

     iconic

     skyline

     and

     busy

     nightlife

    .

     Overall

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     evolve

     and

     develop

    ,

     with

     several

     possible

     trends

     that

     are

     currently

     being

     explored

     and

     researched

    :
    


    1

    .

     Increased

     automation

     and

     robotics

    :

     AI

     is

     becoming

     more

     prevalent

     in

     many

     industries

    ,

     including

     manufacturing

    ,

     transportation

    ,

     and

     service

    .

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     expected

     to

     automate

     more

     and

     more

     tasks

    ,

     potentially

     replacing

     humans

     in

     certain

     jobs

    .
    


    2

    .

     Enhanced

     cognitive

     functions

    :

     AI

     is

     expected

     to

     continue

     to

     improve

     its

     ability

     to

     learn

     and

     adapt

    ,

     potentially

     unlocking

     new

     capabilities

     such

     as

     advanced

     language

     translation

    ,

     emotional

     intelligence

    ,

     and

     advanced

     problem

    -solving

    .
    


    3

    .

     Integration

     of

     AI

     into

     human

     decision

    -making

    :

     AI

     is

     already

     being

     integrated

     into

     many

     decision

    -making

     processes

    ,

     from

     healthcare

    



```python
llm.shutdown()
```
