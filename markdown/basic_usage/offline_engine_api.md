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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.02it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.01it/s]


    2026-05-21 01:27:36,098 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-21 01:27:36] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:44,  1.20it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:44,  1.20it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:44,  1.20it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:44,  1.20it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:44,  1.20it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.28it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.28it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.28it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.28it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  6.28it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  6.28it/s]

    Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:07,  6.28it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03, 10.40it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03, 10.40it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03, 10.40it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03, 10.40it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03, 10.40it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03, 10.40it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:03, 10.40it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:03, 10.40it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:03, 10.40it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 17.19it/s]

    Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:01, 17.19it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 35.14it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 35.14it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 35.14it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 35.14it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 35.14it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 35.14it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 35.14it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 35.14it/s]

    Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 35.14it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 35.14it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 35.14it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:05<00:00, 35.14it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 47.68it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 47.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.37 GB):   3%|▎         | 2/58 [00:00<00:04, 11.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.36 GB):   3%|▎         | 2/58 [00:00<00:04, 11.55it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.20 GB):   3%|▎         | 2/58 [00:00<00:04, 11.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.20 GB):   7%|▋         | 4/58 [00:00<00:04, 12.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.35 GB):   7%|▋         | 4/58 [00:00<00:04, 12.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.34 GB):   7%|▋         | 4/58 [00:00<00:04, 12.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.34 GB):  10%|█         | 6/58 [00:00<00:03, 13.71it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=74.33 GB):  10%|█         | 6/58 [00:00<00:03, 13.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.32 GB):  10%|█         | 6/58 [00:00<00:03, 13.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.32 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.32 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.31 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.25it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.26 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.26 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.24 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.24 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.24 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.24 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.24 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.43it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.24 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.25 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.24 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.24 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.23 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.23 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.21 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.35it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.22 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.35it/s] Capturing num tokens (num_tokens=896 avail_mem=74.21 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.35it/s]Capturing num tokens (num_tokens=896 avail_mem=74.21 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.59it/s]Capturing num tokens (num_tokens=832 avail_mem=74.22 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.59it/s]Capturing num tokens (num_tokens=768 avail_mem=74.23 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.59it/s]Capturing num tokens (num_tokens=704 avail_mem=74.22 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.59it/s]Capturing num tokens (num_tokens=704 avail_mem=74.22 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.50it/s]Capturing num tokens (num_tokens=640 avail_mem=74.22 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.50it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.21 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.50it/s]Capturing num tokens (num_tokens=512 avail_mem=74.19 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.50it/s]Capturing num tokens (num_tokens=512 avail_mem=74.19 GB):  50%|█████     | 29/58 [00:01<00:01, 25.34it/s]Capturing num tokens (num_tokens=480 avail_mem=74.20 GB):  50%|█████     | 29/58 [00:01<00:01, 25.34it/s]Capturing num tokens (num_tokens=448 avail_mem=74.20 GB):  50%|█████     | 29/58 [00:01<00:01, 25.34it/s]Capturing num tokens (num_tokens=416 avail_mem=74.19 GB):  50%|█████     | 29/58 [00:01<00:01, 25.34it/s]Capturing num tokens (num_tokens=416 avail_mem=74.19 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.81it/s]Capturing num tokens (num_tokens=384 avail_mem=74.19 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.81it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.16 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.81it/s]Capturing num tokens (num_tokens=320 avail_mem=74.17 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.81it/s]Capturing num tokens (num_tokens=320 avail_mem=74.17 GB):  60%|██████    | 35/58 [00:01<00:00, 26.03it/s]Capturing num tokens (num_tokens=288 avail_mem=74.16 GB):  60%|██████    | 35/58 [00:01<00:00, 26.03it/s]Capturing num tokens (num_tokens=256 avail_mem=74.16 GB):  60%|██████    | 35/58 [00:01<00:00, 26.03it/s]Capturing num tokens (num_tokens=240 avail_mem=74.13 GB):  60%|██████    | 35/58 [00:01<00:00, 26.03it/s]Capturing num tokens (num_tokens=240 avail_mem=74.13 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.52it/s]Capturing num tokens (num_tokens=224 avail_mem=74.13 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.52it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.12 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.52it/s]Capturing num tokens (num_tokens=192 avail_mem=74.11 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.52it/s]Capturing num tokens (num_tokens=176 avail_mem=74.13 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.52it/s]Capturing num tokens (num_tokens=176 avail_mem=74.13 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.77it/s]Capturing num tokens (num_tokens=160 avail_mem=74.12 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.77it/s]Capturing num tokens (num_tokens=144 avail_mem=74.11 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.77it/s]Capturing num tokens (num_tokens=128 avail_mem=74.11 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.77it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.11 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.15it/s]Capturing num tokens (num_tokens=112 avail_mem=74.10 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.15it/s]Capturing num tokens (num_tokens=96 avail_mem=74.10 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.15it/s] Capturing num tokens (num_tokens=80 avail_mem=74.09 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.15it/s]Capturing num tokens (num_tokens=80 avail_mem=74.09 GB):  83%|████████▎ | 48/58 [00:02<00:00, 28.28it/s]Capturing num tokens (num_tokens=64 avail_mem=74.09 GB):  83%|████████▎ | 48/58 [00:02<00:00, 28.28it/s]Capturing num tokens (num_tokens=48 avail_mem=74.08 GB):  83%|████████▎ | 48/58 [00:02<00:00, 28.28it/s]Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  83%|████████▎ | 48/58 [00:02<00:00, 28.28it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.58it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.58it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.58it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.58it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.54it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:02<00:00, 30.07it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:02<00:00, 24.09it/s]


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
    Generated text:  Nicolette, and I'm a 19-year-old college student who is passionate about science and technology. What is your field of expertise in addition to your interest in science and technology? As a 19-year-old college student, my field of expertise is more specifically in the fields of chemistry and physics. Chemistry focuses on the study of the composition, structure, properties, and reactions of matter, while physics explores the fundamental laws of nature and their application to the physical world. Both areas of study are essential to understanding the world around us and have a wide range of practical applications. I am always eager to learn and to pursue new
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking endorsements from two major political parties. If the president pledges to increase the minimum wage and the other party pledges to lower it, what is the probability that their total endorsement will exceed $3 billion?
    
    To determine the probability that the president's total endorsement will exceed $3 billion, we need to consider the cost of the minimum wage increase for each of the two parties. Let's denote the cost of the minimum wage increase for the first party as \(X\) and for the second party as \(Y\), where \(X\) and \(Y\) are independent and uniformly distributed random variables over the interval \([0, 300
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located in the center of the country, in the Seine river valley. 
    
    Paris is a modern city built on a circular axis with the Seine river as its axis. The center of the city is called the Eiffel Tower, which is the symbol of the city. The city consists of 12 districts, 4 townships and 49 neighborhoods. The district of the Eiffel Tower is located on the center of the city and it is called "Ile de Paris".
    
    Paris is divided into three quarters of the city. The northern quarter of the city is called the "Île de
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, and so is the future of artificial intelligence research. What is our current state of the art? The AI research community is still at an early stage, and we are at a critical juncture for the field. The current state of the art in AI research spans from classical reinforcement learning, deep learning, and statistical learning. New tools and techniques are being developed to explore the boundaries of the field, but the field is still in its infancy. The field of AI research has reached a stage where new advancements are being made, but the field is still in its infancy.
    
    The future of AI research will depend on a variety of factors


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm currently [Current Location] and I enjoy [Favorite Activity or Hobby]. I'm a [Favorite Color] person and I love [Favorite Book, Movie, or Sport]. I'm always [Positive or Negative] about [Something in Your Life]. I'm [Your Name] and I'm [Your Profession]! I'm a [Your Profession] and I'm [Your Profession]! I'm a [Your Profession] and I'm [Your Profession]! I'm a [Your Profession] and I'm [Your Profession]! I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination, known for its rich history, beautiful architecture, and vibrant nightlife. It is a major hub for international business and diplomacy, and is home to many of the world's most prestigious institutions and organizations. The city is also known for its diverse cuisine, including French cuisine, and its role in the French Revolution and the French Revolution. Paris is a city of contrasts,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased integration with other technologies: AI will become more integrated with other technologies such as blockchain, IoT, and quantum computing, leading to new possibilities for AI applications.
    
    2. Personalization and adaptability: AI will become more personalized and adaptable, allowing machines to learn from user behavior and adapt to new situations.
    
    3. Autonomous and self-driving vehicles: AI will become more advanced and autonomous, leading to the development of self-driving cars and other autonomous vehicles.
    
    4. Enhanced security and privacy
    


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
    Generated text:  [Character Name], and I'm a [job title] with [number of years in the industry]. I've always been fascinated by [specific interest or profession], and I've been studying this field for [number of years]. I enjoy [reason for pursuing this career], and I value [benefits or challenges of my current job]. If you're looking for a career in [industry], I'd love to hear from you! 
    
    Please note that I am a character from a fiction book, and my name is [name] but I am not the author. I hope my short introduction helps give you a better sense of who I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France by population and is one of the largest in the world. Paris is known for its rich history, diverse culture, and beautiful architecture. Some of the world's most famous landmarks in Paris include the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its fashion industry, and has produced many famous fashion designers and fashion houses. It has also been home to many important historical and cultural figures such as Napoleon Bonaparte, Victor Hugo, and Gustave Eiffel.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting, with potential applications in virtually every industry and technological field. Here are some potential trends to watch for in the next few decades:
    
    1. Autonomous vehicles: The development of self-driving cars will continue to advance rapidly, with more advanced self-driving technology coming to market. This could transform transportation and healthcare, allowing for more efficient, cost-effective, and environmentally-friendly solutions.
    
    2. Medical imaging: AI has already revolutionized medical imaging, allowing for more accurate diagnosis and treatment of diseases like cancer and heart disease. We may see even more advanced AI that can analyze images and help doctors make faster, more informed decisions.
    
    3. Personalized


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

    ]

     and

     I

     am

     an

     [

    occupation

    ].

     I

     am

     [

    age

    ]

     years

     old

    .

     I

     have

     been

     [

    career

     objective

    ]

     since

     [

    year

     of

     joining

    ]

     and

     have

     successfully

     [

    accom

    pl

    ished

     something

     significant

    ]

     in

     my

     field

    .

     I

     am

     passionate

     about

     [

    a

     personal

     interest

     or

     hobby

    ].

     I

     enjoy

     [

    a

     hobby

     or

     activity

     that

     interests

     me

    ].

     I

     have

     [

    time

     spent

     on

     hobbies

    ]

     and

     have

     been

     [

    status

    ]

     for

     [

    length

     of

     time

    ].

     I

     am

     always

     looking

     for

     new

     experiences

     to

     try

     out

     and

     try

     to

     learn

     something

     new

    .

     I

     am

     excited

     about

     [

    future

     goals

    ]

     and

     dedicated

     to

     achieving

     them

    .

     I

     believe

     in

     [

    a

     personal

     belief

     or

     ethic

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     third

    -largest

     city

     in

     the

     world

     by

     population

    ,

     after

     Beijing

     and

     Shanghai

    .

     The

     city

     is

     known

     for

     its

     historical

     landmarks

    ,

     artistic

     talent

    ,

     and

     vibrant

     culture

    ,

     and

     has

     been

     a

     significant

     center

     of

     French

     politics

    ,

     government

    ,

     and

     industry

     since

     the

     late

     

    1

    9

    th

     century

    .

     The

     city

     is

     also

     famous

     for

     its

     annual

     F

    ête

     de

     la

     Fe

    u

    ille

    ,

     the

     "

    Leaf

     Festival

    ,"

     which

     takes

     place

     in

     September

     and

     features

     a

     variety

     of

     artistic

     activities

    .

     Paris

     is

     home

     to

     the

     Lou

    vre

     Museum

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     E

    iff

    el

     Tower

    ,

     among

     other

     notable

     landmarks

    .

     The

     city

     has

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     vast

     and

     rapidly

     evolving

    ,

     and

     there

     is

     no

     clear

    -cut

     trend

    .

     However

    ,

     here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     Human

     In

    vol

    vement

    :

     AI

     systems

     are

     becoming

     more

     capable

     of

     replic

    ating

     human

     behavior

     and

     decision

    -making

     processes

    ,

     leading

     to

     the

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

     education

    ,

     and

     law

     enforcement

    .

     This

     could

     lead

     to

     a

     more

     human

    -in

    vol

    vement

     AI

    .
    


    2

    .

     AI

     for

     Humans

    :

     AI

     could

     be

     used

     to

     enhance

     human

     capabilities

     in

     areas

     such

     as

     transportation

    ,

     manufacturing

    ,

     and

     healthcare

    .

     For

     example

    ,

     AI

    -powered

     autonomous

     vehicles

     could

     reduce

     accidents

     and

     improve

     traffic

     flow

    ,

     while

     AI

    -powered

     medical

     devices

    



```python
llm.shutdown()
```
