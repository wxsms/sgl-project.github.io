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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.38it/s]


    2026-05-13 14:45:17,455 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 14:45:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.52it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.52it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.52it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.52it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.52it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.52it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.52it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.52it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.52it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.52it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.33it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 38.13it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 38.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.19it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.19it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.19it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 20.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 20.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 20.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 20.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.35it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.35it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.35it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.35it/s] Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.37it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.37it/s]

    Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.37it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.37it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.37it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.37it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=320 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.73it/s]

    Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s]

    Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s] Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  81%|████████  | 47/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.07it/s]

    Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.07it/s] Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.97it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.97it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 37.51it/s]


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
    Generated text:  Ali and I am a PhD student in the Computer Science Department of the University of Manchester. I am interested in both theoretical and applied areas of cryptography and numerical analysis. My research has been focussed on the development of cryptographic algorithms and methods for handling data privacy and protection. I am also interested in developing new computational techniques for solving problems in engineering and physics.
    My thesis, entitled "Two-Step Public Key Cryptosystems", was supervised by Prof. Dr. Hans R. Gerber and was published in the IEEE Transactions on Information Theory. It is a joint work with Dr. Natalia Baranova, Prof. Dr. Marius
    ===============================
    Prompt: The president of the United States is
    Generated text:  from the country of California. What is the name of the country where the president of the United States is from?
    What is the answer?
    A: California is the state where the president of the United States is from. California is the only U.S. state that has a state government, which is a form of democratic government. The president of the United States is elected from that state, and they are considered to be a "state senator" representing the California congressional delegation. However, there are also other states where the president of the United States can be elected, such as New York, Florida, and North Carolina. But California is the
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. London
    C. Berlin
    D. New York
    
    To determine the capital of France, let's consider the following options and examine each one:
    
    A. Paris
    B. London
    C. Berlin
    D. New York
    
    1. **Paris**: Paris is the capital of France. It is an important city with a rich cultural heritage and a beautiful landscape.
    2. **London**: London is the capital of the United Kingdom. While it is an important city, it is not the capital of France.
    3. **Berlin**: Berlin is the capital of Germany. Although it is a large city,
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but the future is also dark. The development of AI is mainly driven by the increase in computing power, but the increase in computing power also brings a problem. For instance, the number of connections between people on the Internet has reached 1.4 billion, and the amount of data has reached 450 petabytes. How to protect personal privacy becomes a big problem. This is a serious issue. The consequences of the Internet data theft will affect various industries, and the social security system will be damaged. Therefore, the protection of personal data in the Internet is becoming very important. Since the Internet is a public network,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Academy of Sciences. Paris is a cultural and economic center with a rich history and a diverse population of over 10 million people. It is also known for its fashion industry, art scene, and annual festivals such as the Eiffel Tower Festival and the Jazz Festival. Paris is a popular tourist destination and a major center for international business and diplomacy. It is the capital of France and a major city in Europe.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: As AI continues to advance, we can expect to see more and more automation in various industries. This could lead to increased efficiency, reduced costs, and improved productivity.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be a growing concern about its ethical implications and the potential for privacy violations. There will be a need for greater regulation and oversight of AI development and deployment.
    
    3. AI for healthcare: AI is
    


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
    Generated text:  [Character's Name], and I'm a [job title] in [company]. I've been working at [company] for [number of years] years now, and I'm really looking forward to what I'll be doing next year. I'm always looking for new challenges and opportunities, and I'm constantly striving to improve myself. What's your name and what's your occupation?
    
    As you can see, you've been introduced to my name and occupation, but I don't have a specific job title or occupation. However, I am a skilled writer and editor, and I have a passion for creating engaging and informative content for a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its vibrant culture, beautiful architecture, and iconic landmarks such as the Eiffel Tower and Louvre Museum. It has a population of around 2.7 million people, and the city is home to many notable institutions and cultural institutions, including the Louvre, the National Library, and the Musée de l'Art et de l'Horreur. Paris is an important cultural hub, hosting numerous world-renowned events and festivals, and its cuisine is also highly acclaimed. The city has a rich and diverse history, and it has been an important center for education and research for centuries.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly dynamic and diverse, and it is likely to experience a wide range of trends and developments in the coming years. Some possible future trends in AI include:
    
    1. Increased privacy and security concerns: With the increasing use of AI in various industries, there is a growing concern about the privacy and security of personal data. This is particularly important for businesses that collect and use data on a massive scale.
    
    2. AI-driven healthcare advancements: AI is already being used in various areas of healthcare, such as diagnosis and treatment planning. As AI technology continues to evolve, it is likely to see further advancements in areas such as personalized medicine, image analysis,


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

     am

     a

     [

    Your

     Profession

    ]

     with

     [

    Your

     Age

    ]

     years

     of

     experience

    .

     I

     am

     a

     [

    Your

     Field

     of

     Study

     or

     Education

    ]

     with

     a

     passion

     for

     [

    Your

     Field

     of

     Study

     or

     Education

    ]

     that

     drives

     me

     to

     always

     strive

     to

     learn

     and

     improve

    .

     I

     am

     [

    Your

     Interest

     or

     Hobby

    ]

     and

     I

     enjoy

     [

    Your

     Interest

     or

     Hobby

    ].

     I

     am

     [

    Your

     Inter

    ests

     or

     Inter

    ests

    ]

     and

     I

     have

     a

     [

    Your

     Hobby

     or

     A

    esthetic

    ].

     I

     believe

     in

     [

    Your

     Professional

     Values

    ]

     and

     I

     am

     always

     looking

     for

     ways

     to

     [

    Your

     Goals

     or

     Goals

    ].

     I

     am

     [

    Your

     Level

     of

     Expert

    ise

     or

     Skills

    ]

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     historical

     and

     cultural

     significance

    ,

     and

     has

     played

     an

     important

     role

     in

     the

     development

     of

     the

     French

     nation

    .

     The

     city

     is

     also

     a

     major

     hub

     for

     international

     trade

     and

     culture

    ,

     and

     has

     a

     rich

     and

     diverse

     local

     culture

    .

     Paris

     is

     home

     to

     many

     famous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Its

     location

     at

     the

     cross

    roads

     of

     Europe

     makes

     it

     a

     popular

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     However

    ,

     the

     city

     has

     faced

     challenges

     in

     recent

     years

    ,

     including

     gent

    r

    ification

     and

     the

     impact

     of

     climate

     change

    .

     Despite

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     wide

     range

     of

     trends

     and

     innovations

    ,

     including

    :
    


     

     

    1

    .

     The

     development

     of

     more

     powerful

     and

     scalable

     artificial

     neural

     networks

    ,

     which

     will

     make

     AI

     more

     accurate

     and

     capable

     of

     performing

     increasingly

     complex

     tasks

    .


     

     

    2

    .

     The

     increasing

     use

     of

     AI

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     energy

    ,

     where

     it

     can

     help

     improve

     efficiency

    ,

     reduce

     costs

    ,

     and

     improve

     safety

    .


     

     

    3

    .

     The

     development

     of

     AI

     that

     can

     learn

     and

     adapt

     to

     new

     situations

     and

     environments

    ,

     rather

     than

     being

     pre

    -program

    med

     or

     rigid

    ly

     structured

    .


     

     

    4

    .

     The

     development

     of

     AI

     that

     can

     operate

     autonom

    ously

     and

     without

     human

     intervention

    ,

    



```python
llm.shutdown()
```
