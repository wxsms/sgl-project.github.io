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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.35it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.35it/s]


    2026-05-14 00:03:25,598 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 00:03:25] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:19,  4.55s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:19,  4.55s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:19,  4.55s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:19,  4.55s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:48,  1.12it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:48,  1.12it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:48,  1.12it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:48,  1.12it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:48,  1.12it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:18,  2.72it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:18,  2.72it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:18,  2.72it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:18,  2.72it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:18,  2.72it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:18,  2.72it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:18,  2.72it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:18,  2.72it/s]Compiling num tokens (num_tokens=2304):  14%|█▍        | 8/58 [00:04<00:18,  2.72it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:04<00:06,  6.98it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=832):  28%|██▊       | 16/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=768):  28%|██▊       | 16/58 [00:04<00:06,  6.98it/s]

    Compiling num tokens (num_tokens=704):  28%|██▊       | 16/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:02, 13.72it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:02, 13.72it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 13.72it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 18.73it/s]

    Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:05<00:01, 18.73it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 25.74it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 25.74it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 25.74it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 25.74it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 25.74it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 25.74it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 25.74it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 25.74it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 25.74it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 33.15it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 33.15it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 33.15it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 33.15it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 33.15it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 33.15it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 33.15it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 33.15it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 33.15it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 33.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 42.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=59.22 GB):   2%|▏         | 1/58 [00:00<00:14,  4.00it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.19 GB):   2%|▏         | 1/58 [00:00<00:14,  4.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.18 GB):   2%|▏         | 1/58 [00:00<00:14,  4.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.18 GB):   5%|▌         | 3/58 [00:00<00:07,  7.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.18 GB):   5%|▌         | 3/58 [00:00<00:07,  7.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.18 GB):   5%|▌         | 3/58 [00:00<00:07,  7.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.18 GB):   9%|▊         | 5/58 [00:00<00:04, 10.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.17 GB):   9%|▊         | 5/58 [00:00<00:04, 10.85it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.16 GB):   9%|▊         | 5/58 [00:00<00:04, 10.85it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.16 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.16 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.31it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=59.16 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.31it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.16 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.31it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.16 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.44it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.15 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.15 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.44it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.15 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.15 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.44it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=59.15 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.14 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.14 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.13 GB):  33%|███▎      | 19/58 [00:01<00:01, 28.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.12 GB):  33%|███▎      | 19/58 [00:01<00:01, 28.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.10 GB):  33%|███▎      | 19/58 [00:01<00:01, 28.58it/s]Capturing num tokens (num_tokens=960 avail_mem=59.12 GB):  33%|███▎      | 19/58 [00:01<00:01, 28.58it/s] Capturing num tokens (num_tokens=896 avail_mem=59.12 GB):  33%|███▎      | 19/58 [00:01<00:01, 28.58it/s]

    Capturing num tokens (num_tokens=832 avail_mem=59.11 GB):  33%|███▎      | 19/58 [00:01<00:01, 28.58it/s]Capturing num tokens (num_tokens=832 avail_mem=59.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.73it/s]Capturing num tokens (num_tokens=768 avail_mem=59.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.73it/s]Capturing num tokens (num_tokens=704 avail_mem=59.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.73it/s]Capturing num tokens (num_tokens=640 avail_mem=59.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.73it/s]Capturing num tokens (num_tokens=576 avail_mem=59.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 32.73it/s]Capturing num tokens (num_tokens=576 avail_mem=59.10 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.62it/s]Capturing num tokens (num_tokens=512 avail_mem=59.09 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.62it/s]Capturing num tokens (num_tokens=480 avail_mem=59.05 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.62it/s]

    Capturing num tokens (num_tokens=448 avail_mem=59.05 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.62it/s]Capturing num tokens (num_tokens=416 avail_mem=59.04 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.62it/s]Capturing num tokens (num_tokens=416 avail_mem=59.04 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.75it/s]Capturing num tokens (num_tokens=384 avail_mem=59.04 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.75it/s]Capturing num tokens (num_tokens=352 avail_mem=59.04 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.75it/s]

    Capturing num tokens (num_tokens=320 avail_mem=59.03 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.75it/s]Capturing num tokens (num_tokens=288 avail_mem=59.03 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.75it/s]Capturing num tokens (num_tokens=288 avail_mem=59.03 GB):  62%|██████▏   | 36/58 [00:01<00:00, 25.62it/s]Capturing num tokens (num_tokens=256 avail_mem=59.03 GB):  62%|██████▏   | 36/58 [00:01<00:00, 25.62it/s]Capturing num tokens (num_tokens=240 avail_mem=59.02 GB):  62%|██████▏   | 36/58 [00:01<00:00, 25.62it/s]Capturing num tokens (num_tokens=224 avail_mem=59.02 GB):  62%|██████▏   | 36/58 [00:01<00:00, 25.62it/s]

    Capturing num tokens (num_tokens=224 avail_mem=59.02 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.25it/s]Capturing num tokens (num_tokens=208 avail_mem=59.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.25it/s]Capturing num tokens (num_tokens=192 avail_mem=59.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.25it/s]Capturing num tokens (num_tokens=176 avail_mem=59.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.25it/s]Capturing num tokens (num_tokens=176 avail_mem=59.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 22.96it/s]Capturing num tokens (num_tokens=160 avail_mem=59.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 22.96it/s]

    Capturing num tokens (num_tokens=144 avail_mem=58.99 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.96it/s]Capturing num tokens (num_tokens=128 avail_mem=58.98 GB):  72%|███████▏  | 42/58 [00:02<00:00, 22.96it/s]Capturing num tokens (num_tokens=128 avail_mem=58.98 GB):  78%|███████▊  | 45/58 [00:02<00:00, 18.17it/s]Capturing num tokens (num_tokens=112 avail_mem=58.98 GB):  78%|███████▊  | 45/58 [00:02<00:00, 18.17it/s]

    Capturing num tokens (num_tokens=96 avail_mem=58.98 GB):  78%|███████▊  | 45/58 [00:02<00:00, 18.17it/s] Capturing num tokens (num_tokens=80 avail_mem=58.48 GB):  78%|███████▊  | 45/58 [00:02<00:00, 18.17it/s]

    Capturing num tokens (num_tokens=80 avail_mem=58.48 GB):  83%|████████▎ | 48/58 [00:02<00:00, 14.82it/s]Capturing num tokens (num_tokens=64 avail_mem=58.39 GB):  83%|████████▎ | 48/58 [00:02<00:00, 14.82it/s]Capturing num tokens (num_tokens=48 avail_mem=58.31 GB):  83%|████████▎ | 48/58 [00:02<00:00, 14.82it/s]Capturing num tokens (num_tokens=32 avail_mem=58.31 GB):  83%|████████▎ | 48/58 [00:02<00:00, 14.82it/s]Capturing num tokens (num_tokens=32 avail_mem=58.31 GB):  88%|████████▊ | 51/58 [00:02<00:00, 16.48it/s]Capturing num tokens (num_tokens=28 avail_mem=58.31 GB):  88%|████████▊ | 51/58 [00:02<00:00, 16.48it/s]Capturing num tokens (num_tokens=24 avail_mem=58.30 GB):  88%|████████▊ | 51/58 [00:02<00:00, 16.48it/s]

    Capturing num tokens (num_tokens=20 avail_mem=58.30 GB):  88%|████████▊ | 51/58 [00:02<00:00, 16.48it/s]Capturing num tokens (num_tokens=20 avail_mem=58.30 GB):  93%|█████████▎| 54/58 [00:02<00:00, 18.63it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:02<00:00, 18.63it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:02<00:00, 18.63it/s]

    Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:02<00:00, 18.63it/s] Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:02<00:00, 17.28it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:02<00:00, 17.28it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:02<00:00, 19.70it/s]


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
    Generated text:  Jason, a 28-year-old high school senior at Kalamazoo College. I enjoy working on the weekends to help with my low self-esteem and depression. I am particularly interested in working with people who have been bullied in school. My main skills include communication skills, social skills, and active listening. I am capable of working on both short and long-term projects. I have experience with a variety of software applications and technology. I have experience with project management, time management, time tracking, and creating budgets. I have also taught my son to drive a car. I have been looking for a new job, and I am
    ===============================
    Prompt: The president of the United States is
    Generated text:  a presidential candidate. He was asked how many children he has and said, "Half of my children's ages add up to 36, and one-third of my children's ages add up to 15." How old is the president of the United States?
    Let's denote the number of children the president of the United States has as \( n \). According to the problem, half of each child's ages add up to 36, and one-third of each child's ages add up to 15. We can express these conditions with the following equations:
    
    1. If we divide the total sum of the ages of
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. What is the capital of the country of which Paris is the capital?
    A. France
    B. Germany
    C. Italy
    D. Spain
    E. Japan
    
    To determine the capital of the country of which Paris is the capital, let's break down the problem step by step.
    
    1. Identify the capital of France: Paris is the capital of France.
    2. Identify the country and its capital: The country is Germany. Germany is the capital of Germany.
    
    Therefore, the capital of the country of which Paris is the capital is Germany.
    
    The correct answer is \boxed{B}.
    ===============================
    Prompt: The future of AI is
    Generated text:  not clear; so is the future of education.
    This week, a state of the art AI system unveiled at the International Conference on Learning Technologies (ICLT), the biggest annual conference of the world’s largest education-focused technology trade show, is proving to be the first of many proof points, as teachers try to figure out how AI can be used in their classrooms. In the past year, teachers have come to see AI as a potential force for good, particularly when it comes to making education more accessible.
    But to use AI to create an inclusive and equitable learning environment, one must first understand the way it’s being used in the classroom,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have a [majority] degree in [major]. I'm a [occupation] who has been working in [industry] for [number of years]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" or simply "Paris". It is the largest city in France and the third-largest city in the world by population. Paris is a cultural, historical, and artistic center, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major financial center and a major transportation hub. Paris is home to many world-renowned museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. It is also a major hub for the French economy
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and privacy. AI developers will need to be more mindful of the potential consequences of their creations and work to ensure that they are developed in a way that is ethical and responsible.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to
    


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
    Generated text:  [Name] and I'm [Age] years old. I'm a [career] with [talent] that has made me famous in [field of interest]. I have a good sense of humor and can make people laugh at [reason for making them famous]. I love [reason for being funny and making people laugh]. My style is [unique skill set]. I'm always up for a challenge, so I'm always looking for new opportunities to make my mark in [field]. What's your background and what's your passion? As an AI language model, I don't have a personal background or feelings, but I'm programmed
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city renowned for its medieval architecture, elegant boulevards, and world-renowned museums. Known as the “City of Light,” it is a cultural center with a thriving arts scene and a diverse population of over 2.7 million people. The city is located on the river Seine and is home to numerous historical landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is a popular destination for tourists and a UNESCO World Heritage site, making it a must-visit destination for travelers visiting the country. The capital of France is known for its unique blend of traditional French culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, with many possibilities and potential applications. Here are some of the most likely future trends:
    
    1. Increased automation and intelligent assistants: AI will continue to advance, and we will see more and more automation and intelligent assistants in our daily lives. These assistants will be able to help us with a wide range of tasks, from managing our finances to providing emotional support.
    
    2. AI for healthcare: AI is already being used in healthcare, with doctors using AI to analyze medical images and make diagnoses. Future AI will likely focus on creating more accurate and efficient healthcare systems, with AI being used to predict disease outbreaks and identify new treatments.
    
    3


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

     [

    Your

     Age

    ]

     years

     old

    .

     I

     am

     passionate

     about

     [

    Your

     passion

     or

     hobby

    ],

     and

     I

    've

     been

     working

     on

     it

     for

     [

    Your

     length

     of

     time

    ]

     years

     now

    .

     I

    've

     been

     learning

     and

     improving

     my

     skills

     and

     techniques

    ,

     and

     I

    'm

     confident

     that

     I

     can

     become

     a

     [

    Your

     ultimate

     goal

     or

     skill

    ]

     in

     the

     future

    .

     I

    'm

     excited

     to

     be

     here

    ,

     and

     I

     look

     forward

     to

     sharing

     my

     knowledge

     and

     experience

     with

     you

    .

     Thanks

     for

     taking

     the

     time

     to

     meet

     me

    .

     #

    Meet

    My

    Name

     #

    Skill

    Share

     #

    Pass

    ion

    For

    Your

    Future

     #

    Self

    Intro

     #

    Skill

    Management

     #

    Skill

    ful

    Persons

     #

    Skill

    ful

    
    


    
    
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

    .

     Its

     cultural

    ,

     economic

    ,

     and

     political

     center

    ,

     Paris

     is

     famous

     for

     its

     historical

     landmarks

    ,

     such

     as

     the

     Lou

    vre

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     Notre

     Dame

     Cathedral

    .

     The

     city

     is

     also

     a

     hub

     for

     fashion

    ,

     art

    ,

     and

     gastr

    onomy

    ,

     and

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     festivals

     throughout

     the

     year

    .

     Paris

     has

     a

     diverse

     population

     with

     over

     

    7

     million

     residents

    ,

     many

     of

     whom

     speak

     French

     and

     English

    .

     Despite

     facing

     challenges

     such

     as

     climate

     change

     and

     economic

     instability

    ,

     Paris

     remains

     a

     vibrant

     and

     influential

     city

     with

     a

     rich

     cultural

     and

     historical

     legacy

    .

     Additionally

    ,

     the

     city

     is

     home

     to

     many

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     full

     of

     exciting

     developments

     and

     changes

    .

     Here

     are

     some

     possible

     trends

     in

     the

     AI

     landscape

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     As

     AI

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

     greater

     personal

    ization

     of

     products

     and

     services

    .

     This

     will

     enable

     businesses

     to

     offer

     more

     tailored

     experiences

     to

     their

     customers

    ,

     leading

     to

     increased

     customer

     satisfaction

     and

     loyalty

    .
    


    2

    .

     AI

     in

     Healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

    ,

     such

     as

     in

     identifying

     the

     best

     drugs

     to

     treat

     diseases

    ,

     and

     in

     diagn

    osing

     and

     treating

     diseases

    .

     As

     AI

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

     more

     innovative

     use

     cases

     in

     healthcare

    ,

     including

     predicting

     the

     likelihood

     of

     diseases

     and

     developing

     personalized

    



```python
llm.shutdown()
```
