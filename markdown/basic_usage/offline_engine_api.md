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
    [2026-04-22 01:50:01] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.06it/s]


    2026-04-22 01:50:05,457 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 01:50:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.76it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.76it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.76it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:30,  1.76it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:30,  1.76it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:11,  4.18it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:11,  4.18it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:11,  4.18it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:11,  4.18it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:03<00:11,  4.18it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:03<00:11,  4.18it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:03<00:11,  4.18it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:03<00:11,  4.18it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:04,  9.57it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:04,  9.57it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:04,  9.57it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:04,  9.57it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:04,  9.57it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:03<00:04,  9.57it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:03<00:04,  9.57it/s]

    Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:03<00:04,  9.57it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:03<00:04,  9.57it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 24.71it/s]

    Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:03<00:01, 24.71it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 33.24it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 33.24it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 33.24it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 33.24it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 33.24it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 33.24it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 33.24it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 33.24it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 33.24it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 33.24it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 43.15it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 43.15it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 43.15it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 43.15it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 43.15it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 43.15it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 43.15it/s]

    Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 43.15it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 43.15it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 43.15it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 43.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.23 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.20 GB):   3%|▎         | 2/58 [00:00<00:04, 13.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:04, 13.06it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:04, 13.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.19 GB):   7%|▋         | 4/58 [00:00<00:04, 12.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.19 GB):   7%|▋         | 4/58 [00:00<00:04, 12.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.19 GB):   7%|▋         | 4/58 [00:00<00:04, 12.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):   7%|▋         | 4/58 [00:00<00:04, 12.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.64it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.19 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.64it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=74.19 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.64it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.18 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.18 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.18 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.25it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.16 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.13 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=960 avail_mem=74.14 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.66it/s] Capturing num tokens (num_tokens=896 avail_mem=74.14 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=832 avail_mem=74.14 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.66it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.13 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=768 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.13it/s]Capturing num tokens (num_tokens=704 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.13it/s]Capturing num tokens (num_tokens=640 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.13it/s]Capturing num tokens (num_tokens=576 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.13it/s]Capturing num tokens (num_tokens=512 avail_mem=74.11 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.13it/s]Capturing num tokens (num_tokens=480 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.13it/s]Capturing num tokens (num_tokens=480 avail_mem=74.13 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=448 avail_mem=74.13 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=416 avail_mem=74.13 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=384 avail_mem=74.12 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.70it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.11 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.70it/s]Capturing num tokens (num_tokens=320 avail_mem=74.11 GB):  60%|██████    | 35/58 [00:01<00:00, 40.06it/s]Capturing num tokens (num_tokens=288 avail_mem=74.11 GB):  60%|██████    | 35/58 [00:01<00:00, 40.06it/s]Capturing num tokens (num_tokens=256 avail_mem=74.11 GB):  60%|██████    | 35/58 [00:01<00:00, 40.06it/s]Capturing num tokens (num_tokens=240 avail_mem=74.11 GB):  60%|██████    | 35/58 [00:01<00:00, 40.06it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  60%|██████    | 35/58 [00:01<00:00, 40.06it/s]Capturing num tokens (num_tokens=208 avail_mem=74.10 GB):  60%|██████    | 35/58 [00:01<00:00, 40.06it/s]Capturing num tokens (num_tokens=208 avail_mem=74.10 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=192 avail_mem=74.10 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=144 avail_mem=74.09 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.83it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.86it/s] Capturing num tokens (num_tokens=80 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.36it/s]Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.36it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.36it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.36it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.36it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.36it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.83it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 35.50it/s]


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
    Generated text:  Prisma and I'm here to discuss our EU-ETC situation and to take questions. You may want to say something. I'm not going to give you time. I'm going to say what my name is and say what I'm here to discuss. So my name is Prisma. My point is to discuss the EU-ETC situation. What do you want to do? You can say something. There is no limit to the number of questions.
    A. That was a real problem. B. I believe in the EU-ETC. C. I think we are overpaying.
    D. I think the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking government officer. According to the constitution, he is elected to office by ________.
    A. The majority of voters in the country
    B. More than half of the voters in the country
    C. The president and the Senate
    D. The people in the country
    
    To determine the correct answer, let's analyze each option step by step:
    
    A. The majority of voters in the country
    - This is not specific to the United States or the presidency. The presidency is filled by the president, not by the majority of the population.
    
    B. More than half of the voters in the country
    - This is a
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. London
    C. New York
    D. Moscow
    Answer:
    A
    
    [Multiple Choice Question] A. Correct B. Incorrect
    What is the premise for police officers to perform police duties? 
    A. Personnel qualification standards
    B. Administrative management
    C. Police equipment and materials
    D. Police training
    Answer:
    Correct
    
    [Multiple Choice Question] Which of the following is an example of a non-redundant software? 
    A. Office 2010
    B. Windows 7
    C. Excel 2010
    D. Photoshop
    Answer:
    
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to be a return to the traditional skills of the physical world.
    
      Reactions
      1. One might argue that the use of AI could displace human workers. In that case, what about a return to the skills of the physical world? 
    
      2. To what extent is AI capable of performing tasks that require human skills?
    
      3. Would the return to the physical world lead to a return to "worlds of the dead"?
    
      4. How would the "return to the physical world" model impact education? 
    
      5. If AI is so advanced that it


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling city with a diverse population and is a major tourist destination. It is home to many famous French artists, writers, and musicians. The city is also known for its cuisine, including its famous croissants and its traditional French dishes. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has been a hub of culture and politics
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    2. Greater integration with human intelligence: AI is already being used to augment human intelligence, such as in virtual assistants and chatbots. As AI becomes more advanced, it may be able to learn and adapt to human behavior, leading to even greater integration with human intelligence.
    
    3
    


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
    Generated text:  [Name] and I am a [fill in the blank with appropriate adjective] [Name], a [fill in the blank with appropriate adjective]. I was born and raised in [city/country], and I have a strong passion for [fill in the blank with appropriate adjective]. Throughout my life, I have always been [fill in the blank with appropriate adjective], [Name] is the person who I strive to be, and [Name] is the person I want to be like. Despite the obstacles that come my way, I always strive to overcome them with [fill in the blank with appropriate adjective] and [fill in the blank
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral. Its history includes a long history of French independence and the rise of the French Revolution, as well as its role in the French Revolution and its status as a major tourist attraction. Paris is also known for its rich cultural heritage, including the Louvre Museum, the Palace of Versailles, and the iconic Eiffel Tower. Finally, Paris is the third most populous city in the world. According to the latest data from the United Nations, Paris is home to around 22 million people. 
    
    The statement is that Paris is known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by significant advancements in several areas. Here are some possible trends to consider:
    
    1. Increased integration with existing technologies: As AI continues to mature, we can expect it to increasingly integrate with various existing technologies, including sensors, cameras, and other devices. This integration could lead to more sophisticated and efficient AI systems that can perform a variety of tasks more effectively and quickly.
    
    2. Increased emphasis on ethical considerations: As AI becomes more integrated into everyday life, there will likely be increased emphasis on ethical considerations, such as ensuring that AI systems do not discriminate based on race, gender, or other factors. This could lead to more


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

    name

    ],

     and

     I

    'm

     a

     dedicated

     [

    job

     title

    ]

     with

     a

     passion

     for

     [

    job

     role

    ].

     I

    'm

     [

    age

    ]

     years

     old

     and

     [

    height

    ]

     inches

     tall

    ,

     with

     [

    weight

    ]

     pounds

    .

     I

    'm

     a

     [

    gender

    ]

     and

     [

    bi

    ological

     sex

    ]

     person

    .

     I

     love

     [

    what

     I

     do

     best

     or

     what

     I

     enjoy

     most

    ].

     I

    've

     always

     been

     [

    born

     into

    ,

     raised

     in

    ,

     or

     adopted

    ],

     and

     I

    'm

     [

    born

     in

    ,

     born

     in

    ,

     or

     born

     to

    ].

     I

    'm

     [

    rel

    atively

    ]

     [

    gener

    ally

    ]

     [

    gener

    ally

    ]

     [

    gener

    ally

    ]

     interested

     in

     [

    what

    's

     important

     to

     me

    ].

     I

    'm

     [

    born

     into

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     of

     love

     and

     light

    .
    


    Why

     should

     I

     learn

     about

     the

     History

     of

     Paris

    ?
    


    Why

     do

     you

     think

     Paris

     is

     the

     capital

     of

     France

    ?
    


    Select

     from

     the

     following

    .


    (I

    )

     It

     is

     the

     capital

     of

     France

    .

     It

     is

     a

     city

     that

     has

     many

     attractions

    .


    (

    II

    )

     It

     is

     the

     capital

     of

     France

    .

     It

     is

     a

     city

     that

     has

     many

     countries

    .


    (

    III

    )

     It

     is

     the

     capital

     of

     France

    .

     It

     is

     a

     city

     that

     has

     many

     beaches

    .


    (

    IV

    )

     It

     is

     the

     capital

     of

     France

    .

     It

     is

     a

     city

     that

     has

     many

     parks

    .

     
    


    I

     think

     the

     answer

     is

    
    


    I

     think

     the

     answer

     is

     (

    I

    )

     It

     is

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     factors

    ,

     including

     advancements

     in

     machine

     learning

    ,

     developments

     in

     neu

    rom

    orphic

     computing

    ,

     and

     increasing

     reliance

     on

     data

     and

     sensor

     technologies

    .

     Here

     are

     some

     potential

     future

     trends

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     other

     areas

     of

     technology

    :

     The

     integration

     of

     AI

     into

     other

     areas

     of

     technology

    ,

     such

     as

     healthcare

    ,

     education

    ,

     and

     transportation

    ,

     is

     likely

     to

     increase

     as

     AI

     becomes

     more

     sophisticated

     and

     widely

     available

    .
    


    2

    .

     More

     automation

     and

     AI

    -powered

     decision

    -making

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     is

     likely

     to

     be

     able

     to

     make

     more

     autonomous

     and

     decision

    -making

     decisions

    ,

     which

     could

     potentially

     reduce

     the

     need

     for

     human

     intervention

     in

     certain

     areas

    .
    


    3

    .

     Greater

    



```python
llm.shutdown()
```
