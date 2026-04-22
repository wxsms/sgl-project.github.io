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
    [2026-04-22 01:00:24] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.04it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.03it/s]


    2026-04-22 01:00:28,867 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 01:00:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 13.86it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 13.86it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 13.86it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 13.86it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.86it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.86it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.86it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 13.86it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 13.86it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 13.86it/s]

    Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:03<00:02, 13.86it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 23.01it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 23.01it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 23.01it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 23.01it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 23.01it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 23.01it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 23.01it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 23.01it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 23.01it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 23.01it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:03<00:01, 23.01it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 33.22it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:03<00:00, 33.22it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 43.61it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 43.61it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 43.61it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 43.61it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 43.61it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 43.61it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 43.61it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 43.61it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 43.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.01it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.02it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.02it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.02it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.14it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.32it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.43it/s]Capturing num tokens (num_tokens=240 avail_mem=76.68 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.43it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.43it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.43it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.43it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.43it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.60it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:01<00:00, 46.90it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 46.90it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 46.90it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 46.90it/s]Capturing num tokens (num_tokens=32 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 46.90it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 46.90it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 30.32it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 30.32it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 30.32it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 30.32it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 30.32it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 30.32it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 35.95it/s]


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
    Generated text:  Mary Ann, and I’m a chemist. My main interest is in the nature of the molecules that make up our world, and how they interact to create the reactions that we see around us every day.
    I first became interested in chemistry after taking a class in high school, and was fascinated by the subject. The professor, who was a true scholar, wanted me to become a chemist. But I didn’t want to be a chemist. I wanted to become a teacher. I loved to teach.
    That’s how my chemistry journey ended. A professor at the University of South Florida had a class to teach chemistry for non-sc
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very busy person. He works hard to make sure that everyone in the country can get what they need. He also works hard to keep people safe and to keep people happy. The president must also be able to understand what the people are saying, and give orders to the army. He must also be able to make sure that he is always safe from any dangers. The president has to work very hard to be the best president. He has to study every day and learn new things. He has to speak with other people and learn how to be a good leader. The president must always be ready to go to work on time. He
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The city is famous for its 23rd century historical monument called Notre Dame Cathedral. Notre Dame Cathedral has a base diameter of 120 meters. If a statue is placed on top of the cathedral, it will be 3.5 meters above the base, and the statue itself will be 2 meters tall. What is the total height of the statue and the cathedral? To find the total height of the statue and the cathedral, we need to consider both the height of the statue and the height of the cathedral. We are given the following information:
    
    1. The base diameter of Notre Dame Cathedral is 1
    ===============================
    Prompt: The future of AI is
    Generated text:  promising, and it has the potential to transform the way we work, learn, and interact with the world around us. But as with any new technology, it is crucial to consider the ethical and social implications of AI. Here are some key considerations:
    
    1. Privacy and Security: AI systems are designed to process and analyze large amounts of data, and they rely on personal information to function. As with any digital system, privacy and security are critical. This means that data should be stored securely and only used for the intended purpose, and that users should be informed of the data they are sharing and how it is being used.
    
    2. Bias


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your personality or background]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new opportunities to grow and learn. What do you do for a living? I'm a [insert a short description of your job role or responsibilities]. I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field. What do you like to do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the country. It is located on the Seine River and is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its rich history, art, and culture, and is a popular tourist destination. The city is also home to many famous landmarks and attractions, including the Louvre Museum, the Champs-Élysées, and the Eiffel Tower. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage. Its status as the capital of France has made it a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we are likely to see more automation and robotics in various industries, from manufacturing to healthcare. This will lead to increased efficiency, cost savings, and job displacement, but it will also create new opportunities for workers.
    
    2. AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI continues to advance, we are likely to see even more applications in healthcare
    


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
    Generated text:  [Name] and I am a [Age] year old male, but I am an adult. I am in my [Job Title] position, and I have been in this field for [X years] of experience. I enjoy exploring new and exciting challenges, and I am always looking for ways to improve my skills and knowledge. I am passionate about [The Character's Passion or Interest] and I am dedicated to being a role model for other adults in the community. I am always learning and growing, and I am committed to staying current with the latest trends and developments in the field. I am a [Personality trait or characteristic
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The statement is: Paris, the capital of France, is renowned for its iconic landmarks such as the Eiffel Tower and the Louvre Museum. Additionally, the city is home to numerous museums, including the Musée d'Orsay and the Musée d'Orsay. Furthermore, Paris has a rich cultural heritage, including a diverse range of museums, theaters, and galleries. Finally, the city is also known for its beautiful architecture, including the iconic Eiffel Tower and its surrounding gardens. 
    
    This statement summarizes the key features of Paris, including its location, historical significance, cultural offerings, architectural beauty, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly unpredictable and constantly evolving, but here are some possible trends that are likely to shape the industry:
    
    1. Integration with human decision-making: As AI becomes more sophisticated, it is becoming more integrated with human decision-making processes. AI algorithms can be used to analyze large amounts of data and make predictions based on that data. This could lead to more effective and efficient decision-making, but also raises ethical concerns about the potential for bias and discrimination.
    
    2. Enhanced personalization: AI can be used to create more personalized experiences for users, based on their behavior and preferences. This could lead to more efficient use of resources and better service delivery, but


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

    age

    ]

     year

     old

     aspiring

     professional

    .

     I

     am

     an

     experienced

     [

    career

     field

    ]

     with

     a

     passion

     for

     [

    mot

    ivation

     or

     hobby

    ].

     I

     love

     to

     [

    exc

    use

     my

     mistake

    ]

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    f

    urther

     my

     skills

    ].

     I

    'm

     a

     [

    occupation

    ]

     and

     I

     have

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     [

    career

     field

    ].

     My

     goal

     is

     to

     [

    specific

     goal

    ],

     and

     I

     believe

     that

     my

     ability

     to

     [

    accom

    pl

    ish

     my

     goal

    ]

     is

     due

     to

     my

     [

    mot

    ivation

    ,

     skill

    set

    ,

     or

     background

    ].

     What

     is

     your

     name

    ,

     and

     what

     do

     you

     do

     for

     a

     living

    ?

    
    
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

     a

     historic

     met

    ropolis

     with

     a

     rich

     cultural

     and

     historical

     significance

    .

     It

     is

     located

     on

     the

     River

     Se

    ine

     and

     is

     a

     major

     center

     for

     the

     arts

    ,

     gastr

    onomy

    ,

     and

     commerce

    .

     The

     city

     also

     hosts

     many

     of

     the

     world

    ’s

     most

     prestigious

     museums

    ,

     theaters

    ,

     and

     events

    ,

     such

     as

     the

     Lou

    vre

    ,

     the

     Palace

     of

     Vers

    ailles

    ,

     and

     the

     Th

    é

    â

    tre

     des

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     is

     also

     the

     birth

    place

     of

     the

     French

     Revolution

     and

     remains

     a

     symbol

     of

     the

     city

    ’s

     cultural

     and

     political

     influence

    .

     The

     city

     is

     also

     known

     for

     its

     exquisite

     architecture

    ,

     such

     as

     the

     Arc

     de

     Tri

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     with

     many

     exciting

     developments

     on

     the

     horizon

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

     customization

     and

     personal

    ization

    :

     With

     AI

    ,

     we

     can

     expect

     to

     see

     more

     personalized

     experiences

     in

     technology

    .

     This

     could

     mean

     machines

     learning

     to

     understand

     and

     respond

     to

     the

     specific

     needs

     and

     preferences

     of

     individual

     users

    ,

     making

     technology

     more

     accessible

     and

     efficient

    .
    


    2

    .

     Improved

     efficiency

     and

     productivity

    :

     As

     AI

     becomes

     more

     powerful

     and

     capable

    ,

     we

     can

     expect

     to

     see

     more

     efficient

     and

     effective

     ways

     to

     perform

     tasks

     and

     complete

     projects

    .

     For

     example

    ,

     AI

    -powered

     automation

     could

     reduce

     the

     time

     and

     resources

     needed

     for

     manual

     tasks

    ,

     freeing

     up

     more

     time

     for

     human

     decision

    -making

     and

     creativity

    .
    


    3

    



```python
llm.shutdown()
```
