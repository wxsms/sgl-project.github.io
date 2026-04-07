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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.25it/s]


    2026-04-07 06:43:37,031 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 06:43:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.84it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.84it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.84it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.84it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.84it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.84it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.84it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.84it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.84it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.84it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.35it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]

    Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:03<00:01, 24.21it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 30.91it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 30.91it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 30.91it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 30.91it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 30.91it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 30.91it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 30.91it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 30.91it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 37.49it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 47.78it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 47.78it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 47.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.12 GB):   2%|▏         | 1/58 [00:00<00:07,  7.58it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.09 GB):   2%|▏         | 1/58 [00:00<00:07,  7.58it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.09 GB):   2%|▏         | 1/58 [00:00<00:07,  7.58it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=76.09 GB):   5%|▌         | 3/58 [00:00<00:04, 12.33it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.09 GB):   5%|▌         | 3/58 [00:00<00:04, 12.33it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.09 GB):   5%|▌         | 3/58 [00:00<00:04, 12.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.08 GB):   5%|▌         | 3/58 [00:00<00:04, 12.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):  10%|█         | 6/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:02, 18.18it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.46it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.46it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.38it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.38it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.38it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.06it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.06it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.27it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  57%|█████▋    | 33/58 [00:00<00:00, 46.86it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 46.86it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 46.86it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 46.86it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:00<00:00, 46.86it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=208 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.35it/s]

    Capturing num tokens (num_tokens=192 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.24it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.24it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.24it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.24it/s]Capturing num tokens (num_tokens=96 avail_mem=74.87 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.24it/s] Capturing num tokens (num_tokens=80 avail_mem=74.87 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.24it/s]Capturing num tokens (num_tokens=64 avail_mem=74.87 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.24it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.87 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.49it/s]Capturing num tokens (num_tokens=48 avail_mem=74.86 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.49it/s]Capturing num tokens (num_tokens=32 avail_mem=74.86 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.49it/s]Capturing num tokens (num_tokens=28 avail_mem=74.85 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.49it/s]Capturing num tokens (num_tokens=24 avail_mem=74.85 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.49it/s]Capturing num tokens (num_tokens=20 avail_mem=74.85 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.49it/s]Capturing num tokens (num_tokens=16 avail_mem=74.85 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.49it/s]Capturing num tokens (num_tokens=16 avail_mem=74.85 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=12 avail_mem=74.84 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=8 avail_mem=74.84 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.09it/s] Capturing num tokens (num_tokens=4 avail_mem=74.84 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.09it/s]Capturing num tokens (num_tokens=4 avail_mem=74.84 GB): 100%|██████████| 58/58 [00:01<00:00, 39.19it/s]


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
    Generated text:  David Bailey, a software engineer with over 10 years of experience in research and development, and I have also held a position as a senior software engineer with the Lockheed Martin Corporation, where I was responsible for developing and maintaining the core software for the DF-21A, the world's first stealth combat aircraft.
    While working for Lockheed Martin I also started to work on my own projects, including starting the project that developed the first security layer, the "Defense Information Positioning System" or DIPS, for the United States Navy. I was also a key member of the development team that was responsible for developing the "Defense Information Processing
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person, not a company. Therefore, the United States does not have companies. The argument's conclusion is flawed because it:
    
    A) Suggests that the United States does not exist
    B) Makes an unsupported claim about the United States
    C) Underscores a conclusion that the United States does not need
    D) Denies the existence of a company
    
    D) Denies the existence of a company
    
    The argument structure is flawed because it suggests that the United States, being a person, does not have companies. This is an unsupported claim about the existence of companies, rather than suggesting that the United States does not exist
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. A．Paris B．Lyon C．Lorient D．Nancy
    
    To determine the capital of France, let's examine each option:
    
    A. Paris - This is the capital of France, which is the largest city and capital of France. It is located in the North-East region of France and is known for its historical landmarks, museums, and the Eiffel Tower.
    
    B. Lyon - Lyon is a city in southern France, known for its historical significance and cultural heritage. It is famous for its Gothic architecture and is part of the Lyon-Perpignan region.
    
    C. Lorient - Lorient is
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of engineers, but what does that mean? AI is the new buzzword in technology that means more than ever before, in fact, it's everywhere. Engineering has changed dramatically since the 1970s, and the engineering workforce is also changing as AI is becoming a bigger part of the technology.
    For those who might not know, the engineering workforce includes not only software developers, but also mechanical engineers, electrical engineers, mechanical engineers, mechanical designers, and other engineers involved in developing and design the technology that is in use today. While some of these roles might have been the same, there is a lot of crossover


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for ways to [job title] and I'm always eager to learn new things. What's your favorite hobby or activity? I'm always looking for new experiences and I'm always eager to try new things. What's your favorite book or movie? I'm always looking for new experiences and I'm always eager
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Parliament building. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to the French Academy of Sciences, the French National Library, and the French Parliament building. Overall, Paris is a vibrant and diverse city with a rich history and culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more transparent and accountable AI systems that are designed to
    


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
    Generated text:  [Name], and I'm a [Age] year old [Occupation], [Position]. I have always been passionate about [Your Passion], and I have been dedicated to learning and growing in my field. I am always looking for new opportunities to help others, and I am always willing to learn from others. My goal is to be [Your Goal], and I am determined to make a positive impact in the world. I am always looking for ways to improve myself, and I am eager to learn and grow. Thank you for taking the time to meet me. [Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a city located on the Seine River in the center of the country. It is home to the French Parliament, the Eiffel Tower, and the Louvre Museum, among other important landmarks. Paris is known for its rich history, art, and architecture, and it continues to be a major cultural and tourist center in France. The city is also known for its nightlife, which attracts millions of visitors each year. Paris has a diverse population of about 1, 678, 543 people and is a major hub of international trade and diplomacy. It is also the seat of the French monarchy and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but there are some potential trends that are likely to shape the technology and its impact on society.
    
    1. Increased automation and productivity: AI will continue to automate tasks that are currently done by humans, such as manufacturing, customer service, and transportation. This will lead to increased productivity and cost savings for businesses.
    
    2. Autonomous vehicles: Self-driving cars and other forms of autonomous vehicle technology will become more common, leading to a reduction in traffic accidents and a decrease in greenhouse gas emissions.
    
    3. Improved medical treatment: AI will be used to develop new treatments for diseases, such as cancer, by analyzing large amounts of medical data.
    
    4


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

     am

     an

     artist

     specializing

     in

     [

    insert

     a

     relevant

     art

     genre

    ,

     such

     as

     painting

    ,

     sculpture

    ,

     or

     illustration

    ].


    I

     am

     a

     highly

     creative

     and

     imaginative

     individual

     with

     a

     passion

     for

     art

     and

     design

    .

     I

     believe

     in

     the

     power

     of

     art

     to

     express

     oneself

     and

     bring

     beauty

     to

     the

     world

    .

     I

     am

     passionate

     about

     using

     my

     art

     to

     make

     a

     difference

     and

     inspire

     others

     with

     my

     creations

    .

     I

     am

     always

     striving

     to

     improve

     my

     skills

     and

     stay

     up

    -to

    -date

     with

     the

     latest

     art

     movements

     and

     techniques

    .

     I

     am

     confident

     in

     my

     abilities

     and

     believe

     that

     my

     artwork

     can

     make

     a

     positive

     impact

     on

     the

     world

    .


    If

     you

     ever

     come

     across

     my

     art

    ,

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     beautiful

     city

     with

     a

     rich

     history

     and

     culture

    ,

     known

     for

     its

     stunning

     architecture

    ,

     elegant

     museums

    ,

     and

     vibrant

     nightlife

    .

     Paris

     is

     also

     a

     popular

     tourist

     destination

    ,

     with

     a

     diverse

     and

     eclectic

     mix

     of

     neighborhoods

     and

     attractions

    .

     It

     is

     the

     political

    ,

     economic

    ,

     and

     cultural

     center

     of

     France

    ,

     and

     its

     status

     as

     a

     global

     met

    ropolis

     is

     a

     testament

     to

     its

     unique

     blend

     of

     traditional

     and

     modern

     elements

    .

     Paris

     is

     also

     a

     symbol

     of

     French

     identity

    ,

     with

     its

     iconic

     landmarks

     and

     historical

     sites

     serving

     as

     a

     source

     of

     pride

     for

     the

     French

     people

    .

     The

     city

     is

     home

     to

     many

     world

    -ren

    owned

     museums

    ,

     including

     the

     Lou

    vre

    ,

     the

     National

     Library

     of

     France

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     range

     of

     different

     trends

     and

     technologies

     that

     are

     expected

     to

     continue

     to

     develop

     and

     evolve

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     that

     could

     influence

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     use

     of

     machine

     learning

    :

     As

     AI

     becomes

     more

     powerful

     and

     accurate

    ,

     it

     is

     likely

     to

     become

     even

     more

     pervasive

     in

     many

     areas

     of

     our

     lives

    .

     Machine

     learning

     will

     become

     even

     more

     integrated

     into

     AI

     systems

    ,

     enabling

     them

     to

     perform

     increasingly

     sophisticated

     tasks

     and

     improving

     their

     ability

     to

     learn

     from

     data

    .
    


    2

    .

     Greater

     focus

     on

     ethical

     AI

    :

     As

     AI

     is

     increasingly

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     an

     increasing

     emphasis

     on

     ensuring

     that

     it

     is

     used

     eth

    



```python
llm.shutdown()
```
