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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.97it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.97it/s]


    2026-05-11 04:24:33,183 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 04:24:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:50,  4.05s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:50,  4.05s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:50,  4.05s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:00,  1.10s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:00,  1.10s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:00,  1.10s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.77it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.77it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.77it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:30,  1.77it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.48it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:07,  6.31it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:07,  6.31it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:07,  6.31it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:07,  6.31it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:07,  6.31it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:04,  9.65it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:04,  9.65it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:04,  9.65it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:04,  9.65it/s]

    Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:04,  9.65it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:04,  9.65it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:04<00:04,  9.65it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:04<00:04,  9.65it/s]Compiling num tokens (num_tokens=832):  28%|██▊       | 16/58 [00:04<00:04,  9.65it/s]Compiling num tokens (num_tokens=768):  28%|██▊       | 16/58 [00:04<00:04,  9.65it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:01, 19.53it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:01, 19.53it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:01, 19.53it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:01, 19.53it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:01, 19.53it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:01, 19.53it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:01, 19.53it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 25.07it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 25.07it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 25.07it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 25.07it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 25.07it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 25.07it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 25.07it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 30.57it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 30.57it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 30.57it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 30.57it/s]

    Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 41.35it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 41.35it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 41.35it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 41.35it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 41.35it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 41.35it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 41.35it/s]

    Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 41.35it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 47.57it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 47.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.19 GB):   2%|▏         | 1/58 [00:00<00:07,  7.15it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.16 GB):   2%|▏         | 1/58 [00:00<00:07,  7.15it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.16 GB):   3%|▎         | 2/58 [00:00<00:07,  7.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.16 GB):   3%|▎         | 2/58 [00:00<00:07,  7.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.16 GB):   3%|▎         | 2/58 [00:00<00:07,  7.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.16 GB):   7%|▋         | 4/58 [00:00<00:05, 10.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.16 GB):   7%|▋         | 4/58 [00:00<00:05, 10.15it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=58.15 GB):   7%|▋         | 4/58 [00:00<00:05, 10.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.15 GB):  10%|█         | 6/58 [00:00<00:05, 10.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.14 GB):  10%|█         | 6/58 [00:00<00:05, 10.14it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=58.14 GB):  10%|█         | 6/58 [00:00<00:05, 10.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.14 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.14 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.13 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.13 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.13 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.90it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.13 GB):  21%|██        | 12/58 [00:01<00:03, 13.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.12 GB):  21%|██        | 12/58 [00:01<00:03, 13.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.12 GB):  21%|██        | 12/58 [00:01<00:03, 13.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.12 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.12 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.50it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=58.11 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.11 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.11 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.11 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.11 GB):  31%|███       | 18/58 [00:01<00:02, 15.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.10 GB):  31%|███       | 18/58 [00:01<00:02, 15.04it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=58.10 GB):  31%|███       | 18/58 [00:01<00:02, 15.04it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.10 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.08it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.08 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.08it/s]Capturing num tokens (num_tokens=960 avail_mem=58.10 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.08it/s] Capturing num tokens (num_tokens=896 avail_mem=58.09 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.08it/s]Capturing num tokens (num_tokens=896 avail_mem=58.09 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.51it/s]Capturing num tokens (num_tokens=832 avail_mem=58.09 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.51it/s]

    Capturing num tokens (num_tokens=768 avail_mem=58.09 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.51it/s]Capturing num tokens (num_tokens=704 avail_mem=58.08 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.51it/s]Capturing num tokens (num_tokens=640 avail_mem=58.08 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.51it/s]Capturing num tokens (num_tokens=640 avail_mem=58.08 GB):  47%|████▋     | 27/58 [00:01<00:01, 23.01it/s]Capturing num tokens (num_tokens=576 avail_mem=58.08 GB):  47%|████▋     | 27/58 [00:01<00:01, 23.01it/s]Capturing num tokens (num_tokens=512 avail_mem=58.07 GB):  47%|████▋     | 27/58 [00:01<00:01, 23.01it/s]Capturing num tokens (num_tokens=480 avail_mem=58.08 GB):  47%|████▋     | 27/58 [00:01<00:01, 23.01it/s]Capturing num tokens (num_tokens=448 avail_mem=58.08 GB):  47%|████▋     | 27/58 [00:01<00:01, 23.01it/s]Capturing num tokens (num_tokens=416 avail_mem=58.08 GB):  47%|████▋     | 27/58 [00:01<00:01, 23.01it/s]Capturing num tokens (num_tokens=416 avail_mem=58.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.12it/s]Capturing num tokens (num_tokens=384 avail_mem=58.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.12it/s]

    Capturing num tokens (num_tokens=352 avail_mem=58.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.12it/s]Capturing num tokens (num_tokens=320 avail_mem=58.06 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.12it/s]Capturing num tokens (num_tokens=288 avail_mem=58.06 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.12it/s]Capturing num tokens (num_tokens=256 avail_mem=57.90 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.12it/s]Capturing num tokens (num_tokens=256 avail_mem=57.90 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.71it/s]Capturing num tokens (num_tokens=240 avail_mem=57.89 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.71it/s]Capturing num tokens (num_tokens=224 avail_mem=57.89 GB):  64%|██████▍   | 37/58 [00:02<00:00, 33.71it/s]Capturing num tokens (num_tokens=208 avail_mem=57.89 GB):  64%|██████▍   | 37/58 [00:02<00:00, 33.71it/s]Capturing num tokens (num_tokens=192 avail_mem=57.89 GB):  64%|██████▍   | 37/58 [00:02<00:00, 33.71it/s]Capturing num tokens (num_tokens=176 avail_mem=57.52 GB):  64%|██████▍   | 37/58 [00:02<00:00, 33.71it/s]

    Capturing num tokens (num_tokens=176 avail_mem=57.52 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.61it/s]Capturing num tokens (num_tokens=160 avail_mem=57.52 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.61it/s]Capturing num tokens (num_tokens=144 avail_mem=57.51 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.61it/s]Capturing num tokens (num_tokens=128 avail_mem=57.33 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.61it/s]Capturing num tokens (num_tokens=112 avail_mem=57.33 GB):  72%|███████▏  | 42/58 [00:02<00:00, 34.61it/s]Capturing num tokens (num_tokens=112 avail_mem=57.33 GB):  79%|███████▉  | 46/58 [00:02<00:00, 30.78it/s]Capturing num tokens (num_tokens=96 avail_mem=57.32 GB):  79%|███████▉  | 46/58 [00:02<00:00, 30.78it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=57.32 GB):  79%|███████▉  | 46/58 [00:02<00:00, 30.78it/s]Capturing num tokens (num_tokens=64 avail_mem=57.32 GB):  79%|███████▉  | 46/58 [00:02<00:00, 30.78it/s]Capturing num tokens (num_tokens=48 avail_mem=57.31 GB):  79%|███████▉  | 46/58 [00:02<00:00, 30.78it/s]Capturing num tokens (num_tokens=48 avail_mem=57.31 GB):  86%|████████▌ | 50/58 [00:02<00:00, 28.82it/s]Capturing num tokens (num_tokens=32 avail_mem=57.31 GB):  86%|████████▌ | 50/58 [00:02<00:00, 28.82it/s]Capturing num tokens (num_tokens=28 avail_mem=57.31 GB):  86%|████████▌ | 50/58 [00:02<00:00, 28.82it/s]

    Capturing num tokens (num_tokens=24 avail_mem=57.30 GB):  86%|████████▌ | 50/58 [00:02<00:00, 28.82it/s]Capturing num tokens (num_tokens=24 avail_mem=57.30 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.02it/s]Capturing num tokens (num_tokens=20 avail_mem=57.30 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.02it/s]Capturing num tokens (num_tokens=16 avail_mem=57.30 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.02it/s]Capturing num tokens (num_tokens=12 avail_mem=57.29 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.02it/s]Capturing num tokens (num_tokens=12 avail_mem=57.29 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.02it/s]Capturing num tokens (num_tokens=8 avail_mem=57.29 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.02it/s] Capturing num tokens (num_tokens=4 avail_mem=57.29 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.02it/s]

    Capturing num tokens (num_tokens=4 avail_mem=57.29 GB): 100%|██████████| 58/58 [00:02<00:00, 21.08it/s]


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
    Generated text:  David. I am a junior at Conestoga High School. I am a three-year player for the team. My favorite team is the Conestoga High School Hornets. I am a member of the Student Council and the Conestoga Student Union. I play basketball on the 4th floor of the school. I am a good swimmer and I have been a member of the varsity swim team for the past two years. I want to be a good teacher and if I can, I want to be a teacher in the future. I want to play basketball and stay with the Conestoga Hornets. I also want to
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to build in a certain country. He knows from previous experience that he will receive n letters a day from the people of that country. The president has decided that for each base, he will receive two new letters a day. How many bases will he be able to establish? (Assume that the addition of new letters to an existing base will not increase the number of new letters he receives from that base).
    To determine how many military bases the president will be able to establish, we need to analyze the information given:
    
    1. The president receives n letters a day from people in the country.
    2.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. What is the capital of Nigeria?
    Answer: Lagos. Lagos is the capital of Nigeria and the largest city in the country.
    Question: Given the information, can we conclude that the statement "Paris is the capital of France" is true?
    Options are:
    A). yes.
    B). it is not possible to tell.
    C). no.
    The answer is A). yes. Paris is the capital of France, and we have the information that Lagos is the capital of Nigeria.
    You are an AI assistant that helps people find information. The person who asked the question below saved the person answers from several different sources so that all information来源
    ===============================
    Prompt: The future of AI is
    Generated text:  here. The next two decades will see a resurgence of AI in many different fields. The field of AI is changing rapidly, and we are entering the era of AI 2.0, where AI systems can be used for a wide variety of purposes, from healthcare to education, and from finance to transportation. The potential impact of AI is truly mind-blowing.
    The world of AI has come a long way since the development of the first computer in the 20th century. In the past 50 years, AI has seen a number of breakthroughs, and the field is now ripe for new growth and innovation. As


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field. I'm always eager to learn and grow, and I'm always looking for opportunities to contribute to the company. What's your favorite hobby or activity? I'm a [job title] at [company name], and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in Europe and the second-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a popular tourist destination and a major economic center in France. It is also home to many important institutions such as the French Academy of Sciences and the French National Library. The city is known for its cuisine, fashion, and music. Paris is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there is a growing emphasis on developing AI that is more ethical and responsible. This could mean developing AI that is designed to minimize harm to individuals or society as a whole, or that is designed to be transparent and accountable.
    
    2. Integration of AI with other technologies: As AI becomes more integrated into other technologies, such as the internet of things (IoT) and the internet of people (IoP
    


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
    Generated text:  [Name]. I'm a [role] who has always had a [weak point or challenge] that has made me [describe how it made you feel, if you need to]. Despite [reason why you've overcome this challenge], I still feel [what you're trying to convey about yourself], and I'm always [express your unique trait or personality trait]. If you'd like to hear more about me, feel free to reach out to me. [And yes, you can include any other details you'd like to include to make your introduction more engaging or interesting]. Welcome to [Name], I hope you find me interesting and inspiring
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city in France, located in the western part of the country and on the River Seine. The city is known for its iconic landmarks like the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum, as well as its rich history and culture. It is also a major transportation hub, with a wide network of roads, railways, and airports that connect the city to other parts of France and the rest of Europe. Paris is home to many famous cultural institutions and a thriving arts and entertainment scene, and its name has become synonymous with sophistication, elegance, and sophistication. The city is also known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be one of unprecedented complexity, innovation, and creativity. Some of the key trends that are expected to shape the field include:
    
    1. Increased focus on ethical AI: As AI becomes more integrated into our lives, there will be increasing pressure to ensure that it is used for the betterment of society. This will likely lead to greater investment in research and development focused on developing AI that is both safe and ethical.
    
    2. Development of more powerful AI systems: As AI continues to evolve and become more complex, there will be a greater emphasis on developing more powerful and flexible AI systems that can perform tasks more efficiently and accurately.
    
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

     __

    ________

    .

     I

     am

     a

     __

    ________

    __

     (

    prefer

    ably

     a

     certain

     type

     of

     character

    ,

     such

     as

     a

     detective

    ,

     scientist

    ,

     or

     superhero

    ).

     I

     am

     a

    /an

     __

    ________

    __

     (

    prefer

    ably

     a

     certain

     profession

    ,

     such

     as

     a

     doctor

    ,

     lawyer

    ,

     or

     engineer

    ).

     My

     name

     is

     __

    ________

    .

     I

     am

     a

    /an

     __

    ________

    __

     (

    prefer

    ably

     a

     certain

     type

     of

     character

    ,

     such

     as

     a

     detective

    ,

     scientist

    ,

     or

     superhero

    ).

     I

     am

     a

    /an

     __

    ________

    __

     (

    prefer

    ably

     a

     certain

     profession

    ,

     such

     as

     a

     doctor

    ,

     lawyer

    ,

     or

     engineer

    ).

     My

     name

     is

     __

    ________

    .

     I

     am

     a

    /an

     __

    ________

    __

     (

    prefer

    ably

     a

     certain

     type

     of

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     in

     the

     center

     of

     the

     country

    .

     It

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

     vibrant

     culture

    ,

     and

     important

     role

     in

     French

     politics

     and

     culture

    .

     Its

     medieval

    城墙

    (

    城墙

    )

     and

     E

    iff

    el

     Tower

     are

     major

     attractions

     and

     a

     popular

     tourist

     destination

    .

     Paris

     is

     also

     a

     major

     economic

     center

    ,

     with

     many

     of

     the

     country

    's

     largest

     companies

     and

     institutions

    .

     Its

     

    8

     million

     inhabitants

     make

     it

     the

     most

     populous

     city

     in

     Europe

    .

     If

     you

    're

     visiting

     Paris

    ,

     be

     sure

     to

     explore

     its

     museums

    ,

     caf

    és

    ,

     and

     art

     galleries

     to

     see

     some

     of

     the

     city

    's

     most

     famous

     landmarks

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     more

     complex

    ,

     collaborative

    ,

     and

     ethical

     approaches

     to

     develop

     and

     deploy

     technology

    .

     Here

     are

     some

     possible

     trends

     that

     may

     be

     observed

     in

     the

     AI

     industry

    :
    


    1

    .

     Enhanced

     AI

     capabilities

    :

     AI

     is

     expected

     to

     become

     even

     more

     powerful

     and

     capable

     in

     the

     coming

     years

    .

     This

     could

     involve

     the

     development

     of

     even

     more

     sophisticated

     algorithms

     that

     can

     learn

     from

     large

     datasets

     and

     make

     more

     accurate

     predictions

    .
    


    2

    .

     Autonomous

     vehicles

    :

     With

     the

     widespread

     adoption

     of

     electric

     and

     autonomous

     vehicles

    ,

     we

     can

     expect

     to

     see

     a

     significant

     increase

     in

     the

     use

     of

     AI

     in

     transportation

    .

     Autonomous

     vehicles

     could

     be

     used

     to

     improve

     traffic

     flow

    ,

     reduce

     accidents

    ,

     and

     increase

     efficiency

    .
    


    3

    .

     Increased

     focus

     on

     ethical

    



```python
llm.shutdown()
```
