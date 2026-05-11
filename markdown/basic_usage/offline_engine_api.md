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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.01it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.00it/s]


    2026-05-11 17:40:52,671 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 17:40:52] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.77it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.37it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.37it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.37it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.37it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.37it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.37it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.37it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.37it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.37it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.37it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 16.37it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.79it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.79it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.79it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.79it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.79it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.79it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.79it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.79it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.79it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.79it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.79it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.88it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:04, 12.48it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:04, 12.48it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:04, 12.48it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):  10%|█         | 6/58 [00:00<00:04, 12.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):  10%|█         | 6/58 [00:00<00:04, 12.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  10%|█         | 6/58 [00:00<00:04, 12.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.50it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.60it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.87it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.87it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.87it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.36it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.36it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.36it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.36it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.52it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.52it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.52it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.52it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.52it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.52it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=288 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.63it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.63it/s]Capturing num tokens (num_tokens=224 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=160 avail_mem=74.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.91it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.91it/s]

    Capturing num tokens (num_tokens=64 avail_mem=73.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=64 avail_mem=73.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=48 avail_mem=74.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=32 avail_mem=74.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=28 avail_mem=74.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=24 avail_mem=74.22 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.98it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.22 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=20 avail_mem=74.22 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.23it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.23it/s]Capturing num tokens (num_tokens=12 avail_mem=74.03 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.23it/s]Capturing num tokens (num_tokens=8 avail_mem=74.03 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.23it/s] Capturing num tokens (num_tokens=4 avail_mem=74.06 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.23it/s]Capturing num tokens (num_tokens=4 avail_mem=74.06 GB): 100%|██████████| 58/58 [00:01<00:00, 30.68it/s]Capturing num tokens (num_tokens=4 avail_mem=74.06 GB): 100%|██████████| 58/58 [00:01<00:00, 29.55it/s]


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
    Generated text:  Tom. I'm from the United States. I'm studying in Beijing now. I have many friends there. We go to the same school. There is a big park in the middle of the school. I like to play on the swings there. Sometimes I play with my friends. I like to play in the playground. Sometimes I play with my friends there. I like to swim in the swimming pool. I like to eat noodles and fish for dinner. I don't like vegetables. I think they taste yucky. How does Tom spend his free time? Tom likes to play with his friends. And he likes to play in the
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected by the members of the legislative branch, and the president of the Senate represents the people of the state of New York.
    The question and answer are below.
    Does the following sentence make sense?
    "The President of the United States is elected by the members of the legislative branch."
    Options:
    - yes
    - no
    Yes.
    The president is elected by members of the legislative branch, not the other way around.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is in the northern part of the country, in the region of the same name. It is located on the right bank of the river Seine. There is also a small island called Jura on the left bank of the river Seine. The center of Paris is the most populous area. It is also the economic center of France. The other areas are small. The central part of Paris is an area called the Latin Quarter. This area is known for its history and architecture. In addition, there are many hotels and restaurants in the center of Paris.
    Summary: Paris is a city in northern France located on the right
    ===============================
    Prompt: The future of AI is
    Generated text:  mostly centered on the development of artificial intelligence, which is also known as AI (Artificial Intelligence). AI involves the creation of artificial intelligence through the use of computer software to perform tasks that are typically done by humans.
    AI is designed to mimic human intelligence and allows computers to learn from data and make decisions based on that data. AI systems can be used in a wide range of applications, including robotics, facial recognition, image and speech recognition, and more.
    The development of AI has been a major focus of research and development in recent years, and it is expected to continue to grow in importance in the coming years. AI has the potential to


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Quarter. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to the French Academy of Sciences and the French Academy of Fine Arts. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the future.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology continues to improve, we can expect to see even more
    


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
    Generated text:  [Name] and I am a [Occupation] from [Location], known for my [Unique Skill or Personality]. I am passionate about [What excites you about what you do] and love to [What you do with your free time]. Whether it's writing, photography, or even something completely unrelated, I am always looking for ways to [What you do that you love to do]. My [Favorite Book/Article/Playlist] is [Title], which truly inspires me to [What you do that you love to do]. I believe that [Why you're a good fit for the job] and am looking forward to [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Given the fact that the answer is "United Nations building", what is the question? The United Nations building is in which city? The United Nations building is located in New York City.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued advancements in areas such as machine learning, natural language processing, computer vision, and robotics. These technologies will enable AI to perform a wide range of tasks, from controlling complex machines to understanding human emotions and language. As AI becomes more integrated into our daily lives, we may see new forms of AI, such as artificial general intelligence, where AI is able to perform tasks that humans and humans do not have the capacity to perform. Additionally, as AI becomes more accessible and affordable, we may see a shift towards more diverse and inclusive applications of AI, including in fields such as healthcare, education, and social welfare. Finally,


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

     [

    Age

    ],

     and

     I

     love

     to

     [

    describe

     a

     hobby

     or

     activity

     that

     interests

     you

    ].

     I

     am

     a

     [

    job

     title

    ]

     who

     have

     been

     [

    number

     of

     years

     in

     this

     position

    ].

     I

     am

     passionate

     about

     [

    describe

     something

     you

     enjoy

     doing

     that

     you

     are

     passionate

     about

    ].

     What

     do

     you

     do

    ?


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

     [

    Age

    ],

     and

     I

     love

     to

     [

    describe

     a

     hobby

     or

     activity

     that

     interests

     you

    ].

     I

     am

     a

     [

    job

     title

    ]

     who

     have

     been

     [

    number

     of

     years

     in

     this

     position

    ].

     I

     am

     passionate

     about

     [

    describe

     something

     you

     enjoy

     doing

     that

     you

     are

     passionate

     about

    ].

     What

     do

     you

     do

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     bustling

     medieval

     neighborhood

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     many

     exciting

     trends

     that

     are

     shaping

     how

     we

     interact

     with

     technology

     and

     how

     it

     can

     be

     used

     to

     benefit

     society

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

     self

    -driving

     technology

    :

     Self

    -driving

     cars

     and

     trucks

     are

     becoming

     more

     common

    ,

     and

     AI

     is

     being

     used

     to

     develop

     safer

     and

     more

     efficient

     self

    -driving

     systems

    .

     These

     systems

     will

     be

     able

     to

     navigate

     roads

    ,

     recognize

     objects

    ,

     and

     make

     decisions

     on

     the

     fly

    ,

     making

     them

     more

     reliable

     and

     efficient

    .
    


    2

    .

     Artificial

     intelligence

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     help

     doctors

     diagnose

     diseases

     and

     develop

     new

     treatments

    .

     As

     AI

     improves

    ,

     we

     may

     see

     more

     personalized

     medicine

     and

     AI

    -supported

     predictive

     analytics

    



```python
llm.shutdown()
```
