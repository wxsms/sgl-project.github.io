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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.30it/s]


    2026-05-10 07:04:34,856 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 07:04:34] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:48,  4.00s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:48,  4.00s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:48,  4.00s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:59,  1.09s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:59,  1.09s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:59,  1.09s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:29,  1.79it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:29,  1.79it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:29,  1.79it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:29,  1.79it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.44it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.44it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.44it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.44it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:14,  3.44it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:14,  3.44it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:14,  3.44it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:05,  7.86it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:05,  7.86it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:05,  7.86it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:05,  7.86it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:05,  7.86it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:05,  7.86it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:05,  7.86it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:05,  7.86it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:02, 14.03it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:02, 14.03it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:02, 14.03it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:02, 14.03it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:02, 14.03it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:02, 14.03it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:02, 14.03it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:02, 14.03it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:02, 14.03it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:02, 14.03it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 23.20it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 23.20it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 23.20it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 23.20it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 23.20it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 23.20it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 23.20it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 23.20it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 23.20it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 23.20it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 23.20it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 34.28it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 34.28it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 34.28it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 34.28it/s]

    Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 34.28it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 34.28it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 34.28it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 34.28it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 34.28it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 34.28it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 43.87it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 43.87it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 43.87it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 43.87it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 43.87it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 43.87it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 43.87it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 43.87it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 43.87it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 43.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.97 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 19.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 19.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 19.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 19.75it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.94 GB):   9%|▊         | 5/58 [00:00<00:02, 22.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.93 GB):   9%|▊         | 5/58 [00:00<00:02, 22.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.92 GB):   9%|▊         | 5/58 [00:00<00:02, 22.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.92 GB):   9%|▊         | 5/58 [00:00<00:02, 22.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.92 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.92 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.91 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.91 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.91 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.45it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.91 GB):  21%|██        | 12/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.90 GB):  21%|██        | 12/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.90 GB):  21%|██        | 12/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.90 GB):  21%|██        | 12/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.89 GB):  21%|██        | 12/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.89 GB):  21%|██        | 12/58 [00:00<00:01, 30.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.89 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.89 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.88 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.88 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.86 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.51it/s]Capturing num tokens (num_tokens=960 avail_mem=72.88 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.51it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=72.88 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.39it/s]Capturing num tokens (num_tokens=896 avail_mem=72.87 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.39it/s]Capturing num tokens (num_tokens=832 avail_mem=72.87 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.39it/s]Capturing num tokens (num_tokens=768 avail_mem=72.87 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.39it/s]Capturing num tokens (num_tokens=704 avail_mem=72.86 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.39it/s]Capturing num tokens (num_tokens=640 avail_mem=72.86 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.39it/s]Capturing num tokens (num_tokens=640 avail_mem=72.86 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.75it/s]Capturing num tokens (num_tokens=576 avail_mem=72.86 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.75it/s]Capturing num tokens (num_tokens=512 avail_mem=72.84 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.75it/s]Capturing num tokens (num_tokens=480 avail_mem=72.86 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.75it/s]Capturing num tokens (num_tokens=448 avail_mem=72.86 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.75it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.86 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.75it/s]Capturing num tokens (num_tokens=416 avail_mem=72.86 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=384 avail_mem=72.85 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=352 avail_mem=72.85 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=320 avail_mem=72.84 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=288 avail_mem=72.84 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=256 avail_mem=72.84 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=256 avail_mem=72.84 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=240 avail_mem=72.83 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=224 avail_mem=72.83 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=208 avail_mem=72.83 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=192 avail_mem=72.83 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.70it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.82 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=176 avail_mem=72.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.06it/s]Capturing num tokens (num_tokens=160 avail_mem=72.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.06it/s]Capturing num tokens (num_tokens=144 avail_mem=72.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.06it/s]Capturing num tokens (num_tokens=128 avail_mem=72.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.06it/s]Capturing num tokens (num_tokens=112 avail_mem=72.81 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.06it/s]Capturing num tokens (num_tokens=96 avail_mem=72.81 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.06it/s] Capturing num tokens (num_tokens=96 avail_mem=72.81 GB):  81%|████████  | 47/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=80 avail_mem=72.81 GB):  81%|████████  | 47/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=64 avail_mem=72.80 GB):  81%|████████  | 47/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=48 avail_mem=72.80 GB):  81%|████████  | 47/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=32 avail_mem=72.80 GB):  81%|████████  | 47/58 [00:01<00:00, 45.89it/s]

    Capturing num tokens (num_tokens=28 avail_mem=72.79 GB):  81%|████████  | 47/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=28 avail_mem=72.79 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.07it/s]Capturing num tokens (num_tokens=24 avail_mem=72.79 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.07it/s]Capturing num tokens (num_tokens=20 avail_mem=72.78 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.07it/s]Capturing num tokens (num_tokens=16 avail_mem=72.78 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.07it/s]Capturing num tokens (num_tokens=12 avail_mem=72.78 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.07it/s]Capturing num tokens (num_tokens=8 avail_mem=72.78 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.07it/s] Capturing num tokens (num_tokens=8 avail_mem=72.78 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=4 avail_mem=72.77 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=4 avail_mem=72.77 GB): 100%|██████████| 58/58 [00:01<00:00, 40.97it/s]


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
    Generated text:  David. My favorite month is October. My favorite reason for being in October is because it's almost the same time as Halloween. Halloween is a place where the excitement and fun are only seen on the first day. October is a time of great change as it transitions from one season to another. It is a time of transforming from summer to winter. It is also a time of great change in the weather and temperature. I don't know about you, but I have seen winter on October first for the first time. This year, it is coming in and it will be very cold. There are many places that are closed and many people
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what government type to adopt. He thinks that a democratic type has more potential for a free society and is less likely to be corrupt. He selects a particular type of democratic government which has only one person in charge of a country. What is the type of government he has chosen? A. Laissez-faire B. Parliamentary C. Presidential D. Aristocratic
    Answer: C
    
    Sodium bicarbonate (NaHCO3) reacts with aqueous ammonia (NH3) to form sodium ammonia (NaONH2) and carbon dioxide (CO2). This reaction belongs to which type of reaction?
    A.
    ===============================
    Prompt: The capital of France is
    Generated text:  a city known as:
    A. Paris
    B. London
    C. Chicago
    D. Tokyo
    The capital of France is Paris. Therefore, the correct answer is A. Paris. London, Chicago, and Tokyo are also cities in France but not its capital. Tokyo is the capital of Japan, which is not France. Chicago is the capital of the United States, not France. London is the capital of the United Kingdom, not France. Therefore, the correct answer is A. Paris. Paris is known for its romantic, historical, and cultural attractions. The other options are not cities in France. Paris is known for its landmarks
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it is not yet a global phenomenon. What we can expect from AI in the next decade?
    We’re in the midst of a period of rapid change. In just a few decades, the amount of data in the world will double. The pace of innovation in AI will grow exponentially.
    While this is exciting news for the future of artificial intelligence, it’s also going to be a bit unsettling. We’re still learning how to use AI. The technology is still in its early stages, and we’re still far from understanding the full scope of its potential. The potential for misuse, the risk of unintended consequences, and the potential


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. Paris is a popular tourist destination and a major hub for business and commerce in Europe. It is also home to many famous museums, including the Louvre and the Musée d'Orsay. The city is known for its vibrant nightlife, fashion, and food scene, and is a popular destination for tourists and locals alike. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine. Additionally, AI is likely to play an increasingly important role in areas such as healthcare, finance, and energy, as it can help to automate and optimize complex processes and systems. However, there are also potential risks and challenges associated with AI, such as the potential for job displacement and the need for ethical and responsible development and use of AI. Overall, the future of AI is likely to be a rapidly evolving
    


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
    Generated text:  [Name], and I’m [Age]. I’m an [Field] expert with a passion for [field], and I’ve always been fascinated by [field] due to its [reason for interest]. I started my [field] career back in [Year], and I’ve always been interested in learning about [field]. I’m always looking for new experiences and learning opportunities, and I’m always eager to expand my knowledge and skills in [field]. I’m passionate about [field] and I’m excited to share my knowledge and experience with others. How about you? What’s your field and what kind of experience do you have?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    -Paris is a major city in France, known for its rich history and culture.
    -It is the largest city in France and the third-largest in the European Union.
    -Paris is the capital of the French Republic, which is the main political and economic hub of France.
    -It is also a cultural center, known for its art, music, and architecture.
    -Paris is renowned for its festivals, museums, and restaurants, attracting millions of tourists annually. 
    
    This statement succinctly captures the key facts about the capital city of France, including its status as the largest and most important city in the country, its cultural significance
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a range of emerging trends, including:
    
    1. Augmented Reality: The rise of AR technology is expected to change the way we interact with our world. It could enable us to see, touch, and interact with objects in the real world in new and exciting ways.
    
    2. Robotics: As robots become more and more intelligent, they are likely to be used in a wider range of applications, from healthcare to manufacturing to transportation.
    
    3. AI Ethics: As AI becomes more integrated into our daily lives, there will be increasing scrutiny of its ethical implications. We will need to develop new ethical frameworks to govern the use


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

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Career

    ],

     living

     [

    Location

    ]

     with

     [

    Person

    ].

     I

    'm

     a

     [

    Appearance

    ],

     [

    D

    iversity

    ],

     and

     [

    Character

    ].

     I

     love

     [

    My

     Skill

     or

     Hobby

    ]

     and

     I

    'm

     passionate

     about

     [

    Interest

    /

    Op

    inion

    ].

     I

    'm

     [

    N

    ost

    alg

    ia

    ],

     [

    M

    ent

    ality

    ],

     and

     [

    Strength

    ].

     I

    'm

     [

    Abs

    ence

     of

     Name

    ],

     but

     I

    'm

     always

     on

     the

     lookout

     for

     new

     adventures

     and

     opportunities

     to

     make

     a

     difference

    .

     Thank

     you

    !

     [

    Name

    ]

     self

    -int

    rodu

    ces

     themselves

     and

     provides

     a

     brief

     background

     about

     themselves

    ,

     including

     their

     career

    ,

     location

    ,

     appearance

    ,

     personality

    ,

     skills

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

     and

     a

     wide

     range

     of

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    ’

    Or

    say

    .

     The

     city

     is

     also

     famous

     for

     its

     elaborate

     music

     and

     dance

     scene

    ,

     including

     the

     famous

     É

    to

    iles

     de

     Paris

    .

     Paris

     is

     home

     to

     numerous

     international

     festivals

     and

     events

     throughout

     the

     year

    ,

     including

     the

     famous

     E

    iff

    el

     Tower

     Festival

     in

     May

    .

     It

     is

     also

     known

     for

     its

     rich

     culinary

     scene

    ,

     with

     famous

     dishes

     such

     as

     cro

    issant

     and

     esc

    arg

    ot

    .

     Paris

     is

     a

     vibrant

     and

     bustling

     city

     with

     a

     strong

     sense

     of

     French

     culture

     and

     identity

    .

     Its

     history

    ,

     architecture

    ,

     and

     food

     offerings

     have

     made

     it

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

    ,

     including

    :
    


     

     

    1

    .

     Increased

     focus

     on

     ethical

     and

     social

     implications

     of

     AI

    .

     As

     more

     people

     become

     interested

     in

     the

     impact

     of

     AI

     on

     society

    ,

     there

     may

     be

     greater

     emphasis

     on

     addressing

     ethical

     concerns

     and

     balancing

     the

     benefits

     and

     risks

     of

     AI

    .


     

     

    2

    .

     Advances

     in

     AI

     technology

     and

     software

     that

     make

     it

     more

     accessible

     and

     user

    -friendly

    .

     As

     AI

     becomes

     more

     widely

     adopted

    ,

     there

     may

     be

     an

     increased

     need

     for

     tools

     and

     software

     that

     make

     AI

     easier

     to

     use

     and

     less

     intimidating

     for

     users

    .


     

     

    3

    .

     Integration

     with

     human

     emotions

     and

     perspectives

    .

     As

     AI

     becomes

     more

     sophisticated

    ,

     there

     may

     be

     an

     increased

     focus

     on

    



```python
llm.shutdown()
```
