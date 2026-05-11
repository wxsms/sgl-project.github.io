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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.25it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.25it/s]


    2026-05-11 08:21:01,992 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 08:21:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:07,  6.01it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03, 10.17it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:03, 10.17it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:02, 15.13it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:02, 15.13it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:04<00:02, 15.13it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:02, 15.13it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:04<00:02, 15.13it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:04<00:02, 15.13it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 19.30it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 19.30it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 19.30it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 19.30it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 19.30it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 19.30it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 19.30it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 25.25it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 25.25it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 25.25it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 25.25it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 25.25it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 25.25it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 25.25it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 25.25it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 25.25it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 34.10it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 34.10it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 34.10it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 34.10it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 34.10it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 34.10it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 34.10it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 34.10it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 34.10it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 42.63it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 42.63it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 42.63it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 42.63it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 42.63it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 42.63it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.68 GB):   3%|▎         | 2/58 [00:00<00:03, 15.41it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.69 GB):   3%|▎         | 2/58 [00:00<00:03, 15.41it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.69 GB):   3%|▎         | 2/58 [00:00<00:03, 15.41it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=69.69 GB):   7%|▋         | 4/58 [00:00<00:03, 17.34it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.68 GB):   7%|▋         | 4/58 [00:00<00:03, 17.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.67 GB):   7%|▋         | 4/58 [00:00<00:03, 17.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.66 GB):   7%|▋         | 4/58 [00:00<00:03, 17.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.66 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.90it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=69.64 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.64 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.63 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.63 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.62 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=69.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.59 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=69.59 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=69.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=960 avail_mem=69.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.74it/s] Capturing num tokens (num_tokens=896 avail_mem=69.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=832 avail_mem=69.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=832 avail_mem=69.58 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.39it/s]Capturing num tokens (num_tokens=768 avail_mem=69.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.39it/s]Capturing num tokens (num_tokens=704 avail_mem=69.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.39it/s]

    Capturing num tokens (num_tokens=640 avail_mem=69.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.39it/s]Capturing num tokens (num_tokens=576 avail_mem=69.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.39it/s]Capturing num tokens (num_tokens=512 avail_mem=69.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.39it/s]Capturing num tokens (num_tokens=512 avail_mem=69.55 GB):  50%|█████     | 29/58 [00:00<00:00, 40.61it/s]Capturing num tokens (num_tokens=480 avail_mem=69.57 GB):  50%|█████     | 29/58 [00:00<00:00, 40.61it/s]Capturing num tokens (num_tokens=448 avail_mem=69.57 GB):  50%|█████     | 29/58 [00:00<00:00, 40.61it/s]Capturing num tokens (num_tokens=416 avail_mem=69.57 GB):  50%|█████     | 29/58 [00:00<00:00, 40.61it/s]Capturing num tokens (num_tokens=384 avail_mem=69.56 GB):  50%|█████     | 29/58 [00:00<00:00, 40.61it/s]Capturing num tokens (num_tokens=352 avail_mem=69.56 GB):  50%|█████     | 29/58 [00:00<00:00, 40.61it/s]Capturing num tokens (num_tokens=352 avail_mem=69.56 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=320 avail_mem=69.55 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=288 avail_mem=69.55 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.84it/s]

    Capturing num tokens (num_tokens=256 avail_mem=69.55 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=240 avail_mem=69.54 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=224 avail_mem=69.54 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=224 avail_mem=69.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=208 avail_mem=69.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=192 avail_mem=69.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=176 avail_mem=69.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=160 avail_mem=69.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=144 avail_mem=69.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=144 avail_mem=69.53 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=128 avail_mem=69.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=112 avail_mem=69.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.60it/s]

    Capturing num tokens (num_tokens=96 avail_mem=69.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.60it/s] Capturing num tokens (num_tokens=80 avail_mem=69.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=64 avail_mem=69.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=64 avail_mem=69.51 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=48 avail_mem=69.51 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=32 avail_mem=69.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=28 avail_mem=69.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=24 avail_mem=69.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=20 avail_mem=69.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=20 avail_mem=69.49 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.91it/s]Capturing num tokens (num_tokens=16 avail_mem=69.49 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.91it/s]Capturing num tokens (num_tokens=12 avail_mem=69.49 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.91it/s]

    Capturing num tokens (num_tokens=8 avail_mem=69.48 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.91it/s] Capturing num tokens (num_tokens=4 avail_mem=69.48 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.91it/s]Capturing num tokens (num_tokens=4 avail_mem=69.48 GB): 100%|██████████| 58/58 [00:01<00:00, 37.91it/s]


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
    Generated text:  Victor.
    I’m a brand new member of the community.
    I’m not a member of this community for work or personal reasons. I’m just a random person who just stumbled upon this place.
    I’m curious to know how to do something related to coding, so I’m not sure where to start.
    I’ve heard of React and Vue.js, but I’m not sure which one to use. Do you have any advice on which one to pick?
    Certainly! Choosing between React and Vue.js can be a bit of a heads up, but both are powerful JavaScript frameworks that are quite popular among developers. Let’s dive into some of the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she is responsible for the country. In fact, the president is very important to the country. The president is like the leader of the country. He or she has to be very strict with the people. He or she is always ready to do something to improve the country. The president is very important because the country is very important. He or she is responsible for the country and the people. He or she has to make sure that the country is safe. The president is responsible for the country and the people. He or she has to make sure that the country is free of drugs and crime. The
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.  Given a list of categories: company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, landmark, organization, park, river, species, food, place of learning, special interest group. What category does the paragraph belong to?
    A:
    To determine the category that the paragraph belongs to, I will analyze the key information provided:
    
    1. The paragraph is about the capital of France.
    2. It mentions that Paris is the capital.
    3. The paragraph focuses on the specific structure of Paris, not its functions, characteristics, or general characteristics.
    
    Based on this analysis, the category
    ===============================
    Prompt: The future of AI is
    Generated text:  in high-tech and not in the lab
    
    The future of AI is in high-tech and not in the lab
    
    I'm a big believer in the idea that AI will be a huge part of how we interact with the world. It's one of the most exciting things in the technology world right now.
    
    But it is not going to come from the lab. It's coming from high tech, and it's coming from all over the world.
    
    Right now, the people designing AI are mostly from the US. That's because technology and AI are more common in the US, and more people have access to it. But the people doing the


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast. I'm passionate about [What interests you about your occupation]. I'm always looking for new challenges and opportunities to grow and learn. I'm always eager to learn and improve myself. I'm a [What is your favorite hobby or activity] and I enjoy [What makes you happy]. I'm a [What is your favorite book or movie] and I love [What makes you excited about the book/movie]. I'm a [What is your favorite person or thing] and I love [What makes you feel
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is known for its fashion, art, and cuisine, and is a major economic center in Europe. It is also home to many famous landmarks and attractions, including the Louvre, the Eiffel Tower, and the Notre-Dame Cathedral. Paris is a city that has a unique blend of history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations and responsible use of AI. This could lead to more stringent regulations and guidelines for AI development and
    


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
    Generated text:  [Name], I'm a [Majority] computer programmer and [Minority] game designer. I'm [Age] years old, [Degree] from [School], and [Profession]. I'm [特长] with [Tech Skill] and [Favorite Hobby]. I enjoy [Favorite Activity], and [Project] projects that I've done in the past, which have helped me grow and improve. I'm [Stage of Life], with my [Height] feet and [Weight] pounds, and I'm [Stomach]. My [Favorite Movie] is [Title], and my [Favorite Book] is [Title],
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the "City of Love" due to its rich cultural and historical significance. It is located on the Seine River and has been a major transportation hub for centuries. The city is home to iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is known for its vibrant nightlife, romantic architecture, and bustling crowds. It is also known for its unique culinary scene, with its famous dishes such as foie gras and escargot. Paris is a major tourist destination and a UNESCO World Heritage site, making it a must-visit city for anyone traveling to France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a variety of factors, including developments in quantum computing, machine learning, natural language processing, and robotics. Some potential future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more sophisticated, there will be increasing pressure to ensure they are used for beneficial purposes and do not cause harm. This could lead to greater attention to ethical considerations and regulations around AI.
    
    2. Development of more advanced natural language processing: The ability to understand and interpret human language will be crucial as AI systems become more widespread. This could lead to advancements in areas such as machine translation, sentiment analysis, and question-


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

     who

     is

     here

     to

     assist

     and

     help

     anyone

     who

     needs

     it

    .

     From

     the

     moment

     I

     walk

     through

     the

     door

    ,

     I

    'm

     here

     to

     serve

     and

     ensure

     your

     safety

     and

     comfort

    .

     If

     you

     need

     anything

    ,

     just

     let

     me

     know

    ,

     and

     I

    'll

     be

     here

     to

     make

     sure

     everything

     goes

     smoothly

    .

     Whether

     it

    's

     a

     problem

     with

     your

     medication

    ,

     a

     sudden

     illness

    ,

     or

     just

     a

     moment

     of

     frustration

    ,

     I

    'll

     be

     here

     to listen

     and

     help

     you

     feel

     better

    .

     So

    ,

     please

     feel

     free

     to

     reach

     out

     to

     me

     anytime

    ,

     and

     I

    'll

     be

     here

     to

     help

    .

     [

    Your

     Name

    ]

     is

     a

     [

    Your

     Profession

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     and

     most

     populous

     city

     in

     France

    ,

     with

     an

     estimated

     population

     of

     

    2

    .

     

    4

     million

     people

     as

     of

     

    2

    0

    2

    1

    .

     It

     is

     the

     capital

     of

     the

     French

     Department

     of

     Paris

    ,

     which

     also

     includes

     nearby

     regions

     such

     as

     the

     Î

    le

    -de

    -F

    rance

     and

     Î

    le

    -de

    -O

    t

    zy

    x

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     cultural

     scene

    .

     It

     is

     also

     one

     of

     the

     most

     cosm

    opolitan

     cities

     in

     Europe

     and

     is

     home

     to

     many

     of

     the

     country

    's

     political

    ,

     cultural

    ,

     and

     economic

     centers

    .

     Paris

     is

     a

     symbol

     of

     France

     and

     a

     major

     cultural

     and

     tourist

     destination

    .

     In

     addition

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     diverse

     and

     constantly

     evolving

    ,

     with

     new

     possibilities

     emerging

     every

     day

    .

     Some

     of

     the

     most

     promising

     trends

     include

    :
    


    1

    .

     Improved

     AI

     ethics

     and

     accountability

    :

     The

     ethical

     use

     of

     AI

     is

     becoming

     a

     critical

     concern

     as

     concerns

     over

     privacy

    ,

     bias

    ,

     and

     accountability

     grow

    .

     AI

     systems

     need

     to

     be

     designed

     and

     trained

     in

     a

     way

     that

     respects

     human

     values

     and

     avoids

     harm

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     diagnose

     diseases

    ,

     predict

     patient

     outcomes

    ,

     and

     assist

     in

     treatment

     planning

    .

     As

     the

     technology

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     even

     more

     widespread

     use

     in

     healthcare

    .
    


    3

    .

     AI

     in

     manufacturing

    :

     AI

     is

     being

     integrated

     into

    



```python
llm.shutdown()
```
