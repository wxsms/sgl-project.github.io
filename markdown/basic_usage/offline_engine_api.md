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
    [2026-04-25 15:50:23] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.98it/s]


    2026-04-25 15:50:27,842 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-25 15:50:27] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.47it/s]

    Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 29.47it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 37.24it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 37.24it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 37.24it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 37.24it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 37.24it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 37.24it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 37.24it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 37.24it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 37.24it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 37.24it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 45.68it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 45.68it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 45.68it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 45.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.34it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.37 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.90it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  21%|██        | 12/58 [00:00<00:01, 29.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 29.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 29.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 29.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  21%|██        | 12/58 [00:00<00:01, 29.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  21%|██        | 12/58 [00:00<00:01, 29.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.03it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.03it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.03it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.03it/s] Capturing num tokens (num_tokens=960 avail_mem=137.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.49it/s]Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.49it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.49it/s]Capturing num tokens (num_tokens=768 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.49it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.49it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.49it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=576 avail_mem=137.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.90it/s]

    Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.01it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.01it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.01it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.01it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]

    Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=176 avail_mem=137.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=176 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.28it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.28it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.28it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.28it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.28it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.28it/s] Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  81%|████████  | 47/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  81%|████████  | 47/58 [00:01<00:00, 40.34it/s]

    Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  81%|████████  | 47/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  81%|████████  | 47/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  81%|████████  | 47/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  81%|████████  | 47/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.48it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.48it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.48it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.48it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.48it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.48it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.90it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.90it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 37.05it/s]


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
    Generated text:  Gracie and I am 17 years old. I am from South Africa. I am an aspiring chef. I have a passion for cooking and taking care of my family and friends. Cooking is my way of spreading love and happiness to the people around me. I am proud to be a part of the South African community. I am a volunteer at a local community center, helping out with food preparation, cleaning, and outreach. I have also been an intern at a local restaurant, learning new skills and gaining experience in the food industry. I have a crush on a boyfriend named Jake, and we have been dating for two years
    ===============================
    Prompt: The president of the United States is
    Generated text:  4 feet tall. If it's a certain event day, the president is 10% taller than usual. If it's a general event day, he is 25% taller than usual. How tall is the president on a general event day?
    
    To determine the height of the president on a general event day, we need to calculate 25% of his usual height and add it to his usual height.
    
    The president's usual height is 4 feet.
    
    First, we calculate 25% of 4 feet:
    \[ 25\% = \frac{25}{100} = 
    ===============================
    Prompt: The capital of France is
    Generated text:  _____. ____
    A. Paris
    B. Moscow
    C. Shanghai
    D. Kuala Lumpur
    Answer: A
    
    When the X-ray shows calcification of the peripheral lung region, what should be excluded?
    A. Chronic lung abscess
    B. Chronic atelectasis
    C. Tuberculoma
    D. Pulmonary bulla
    E. Chronic fibrocavitary pulmonary tuberculosis
    Answer: D
    
    After the end of the Anti-Japanese War, the leaders of the Kuomintang announced the end of the war through a statement issued in December 1945. This statement, which is referred to as
    ===============================
    Prompt: The future of AI is
    Generated text:  multifaceted, but one of the most promising areas is in the field of natural language processing (NLP). NLP is the study of how computers process and interpret human language, and it has been applied to a variety of tasks including text generation, speech recognition, and natural language generation.
    
    One of the key challenges in NLP is the processing and understanding of large amounts of textual data. This is often referred to as the "text generation" task, as it involves generating new text based on the content of existing text. The challenge is that the amount of text data available to NLP models can be very large and complex, which


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [job title]. I'm a [job title] at [company name] and I'm always looking for ways to [describe your job role]. I'm a [job title] at [company name] and I'm always looking for ways to [describe your job role]. I'm a [job title] at [company name] and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and restaurants. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French language. It is also home to many notable artists, writers, and musicians. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also known for its cuisine, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is likely to become more integrated into various industries, leading to increased automation of tasks and processes. This could result in the creation of new jobs, but also potentially leading to job displacement.
    
    2. AI ethics and privacy concerns: As AI becomes more integrated into our daily lives, there will be increasing concerns about its ethical implications and potential privacy violations. This could lead to further regulation and development of new ethical guidelines for AI.
    
    3. AI-driven healthcare: AI is already being used in healthcare to diagnose and treat diseases, but there is potential for even more advanced AI-driven
    


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
    Generated text:  [Name]. I am a [occupation] with [number of years] years of experience. I am excited to work with you today and assist you in [what you would like to accomplish]. Please let me know what you would like from me.
    Name: [Enter Character's Name]
    Occupation: [Enter Character's Occupation]
    Number of Years: [Enter Character's Number of Years]
    Experience: [Enter Character's Experience] Hello, my name is [Name]. I am a [occupation] with [number of years] years of experience. I am excited to work with you today and assist you in [what you would like
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Factual statement: The capital of France is Paris. 
    
    This is a concise factual statement that directly and clearly states the facts about the capital city of France. It does not contain any additional information or context that would make it longer or more detailed. The statement itself is a simple and accurate representation of the facts about the French capital. 
    
    Here's an expanded version of the statement for clarity:
    
    "The capital city of France is Paris. It is the largest city in France and the largest city in Europe by population, located in the Hauts-de-France region of the central-western part of the country. Paris is known
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a combination of rapid innovation, technological advancements, and changing societal needs. Here are some possible future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see more widespread automation and robotics in various industries, such as manufacturing, transportation, and service.
    
    2. Enhanced privacy and security: With the increasing use of AI in daily life, there is a growing concern about the potential misuse of AI technology. As AI technology becomes more advanced and complex, we can expect to see increased regulations and privacy protections to ensure that AI systems are used ethically and responsibly.
    
    3.


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

    ]

     and

     I

     am

     a

     [

    Your

     Profession

    ]

     with

     a

     passion

     for

     [

    Your

     Hobby

    /

    Interest

    /

    Adv

    oc

    acy

    ].

     I

     enjoy

     sharing

     my

     knowledge

     and

     experiences

     with

     others

     through

     my

     writing

    ,

     speaking

    ,

     and

     my

     role

     in

     advocating

     for

     social

     justice

     and

     equality

    .

     I

     am

     excited

     to

     learn

     more

     about

     you

     and

     how

     we

     can

     make

     a

     positive

     impact

     together

    .

     How

     can

     I

     get

     to

     know

     you

     better

    ?

     Please

     describe

     how

     you

     would

     approach

     the

     conversation

    .

     Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ]

     and

     I

     am

     a

     [

    Your

     Profession

    ]

     with

     a

     passion

     for

     [

    Your

     Hobby

    /

    Interest

    /

    Adv

    oc

    acy

    ].

     I

     enjoy

     sharing

     my

     knowledge

     and

     experiences

     with

     others

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    What

     is

     the

     answer

    ?

     Paris

     is

     the

     capital

     of

     France

    .

     What

     is

     the

     answer

    ?

     Paris

     is

     the

     capital

     of

     France

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

     and

     there

     are

     numerous

     potential

     trends

     that

     could

     shape

     the

     direction

     of

     the

     technology

    .

     Here

     are

     some

     of

     the

     most

     promising

     areas

     for

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increase

     in

     the

     use

     of

     AI

     in

     healthcare

    :

     AI

     has

     the

     potential

     to

     revolution

    ize

     healthcare

     by

     improving

     diagnostic

     accuracy

    ,

     personalized

     treatment

     plans

    ,

     and

     reducing

     errors

    .

     For

     example

    ,

     AI

     could

     be

     used

     to

     predict

     which

     patients

     are

     at

     risk

     of

     developing

     diseases

     or

     complications

    ,

     allowing

     healthcare

     providers

     to

     intervene

     early

     to

     prevent

     disease

     progression

    .
    


    2

    .

     Enhanced

     cognitive

     function

     in

     everyday

     tasks

    :

     AI

     is

     already

     making

     progress

     in

     areas

     like

     language

     translation

    ,

     image

     recognition

    ,

     and

     speech

     synthesis

    .

     As

     the

     technology

     continues

     to

     evolve

    



```python
llm.shutdown()
```
