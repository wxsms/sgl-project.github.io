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


    2026-04-07 20:15:45.439 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 20:15:45] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 20:15:45.439 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 20:15:45] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 20:15:45.439 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 20:15:45] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 20:15:45.439 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 20:15:45] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 20:15:45.439 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 20:15:45] Persistent cache disabled, using in-memory JIT cache


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.36it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.35it/s]


    2026-04-07 20:15:47,846 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 20:15:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.17it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.17it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.17it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.17it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.17it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.17it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 13.17it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 13.17it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:02, 13.17it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 27.97it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 27.97it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 27.97it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 27.97it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 27.97it/s]

    Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 27.97it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 27.97it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 27.97it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 34.14it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 34.14it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 34.14it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 34.14it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 34.14it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 34.14it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 34.14it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 34.14it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 40.04it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 40.04it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 40.04it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 40.04it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 40.04it/s]

    Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 40.04it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 40.04it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 40.04it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 40.04it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 40.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   2%|▏         | 1/58 [00:00<00:06,  9.25it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   2%|▏         | 1/58 [00:00<00:06,  9.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   2%|▏         | 1/58 [00:00<00:06,  9.25it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   2%|▏         | 1/58 [00:00<00:06,  9.25it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 17.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 17.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 17.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 17.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.00it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.51it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.51it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.51it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.50it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.50it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.50it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.50it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.50it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.50it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.70it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.70it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.70it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.70it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.70it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.70it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.40it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.40it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.40it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.40it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.40it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.40it/s]Capturing num tokens (num_tokens=240 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.40it/s]Capturing num tokens (num_tokens=240 avail_mem=76.68 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.45it/s]

    Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.63it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.63it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.63it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.63it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.63it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.63it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.63it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.56it/s]Capturing num tokens (num_tokens=32 avail_mem=76.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.56it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.56it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.56it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.56it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.56it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.56it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.61it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.61it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.61it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 41.77it/s]


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
    Generated text:  Rezhi and I am a full-stack developer focused on digital marketing. I am an entrepreneur at the forefront of digital marketing and a professional in the field of software. I started my career as a software developer in the field of Windows software development. I have been a software developer since I was 22 years old. I have extensive experience in the field of software development and have been involved in multiple projects, including web and mobile applications, desktop and server applications, and mobile app development.
    I have been working in the field of digital marketing for the past 13 years. I have been a digital marketing consultant, web design consultant
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. But do you know what does he do? The president usually works long hours. He spends all day in the White House. But do you know how he goes to the White House? Let me tell you in a little more detail. First, he has to walk to his home. And then he has to drive to his office. Now, I know some people drive to the White House. But not many people are very careful when driving to the White House. Can you imagine how fast the car is in the White House? The president drives very fast. It's much faster than a car. He has to
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Among the following places, the one that would be the capital of France is (     )
    A: Sydney
    B: Stockholm
    C: Athens
    D: London To determine which city would be the capital of France, we need to understand the structure of the country's political system. France is a constitutional monarchy, meaning the king or queen acts as the head of state, while the government is led by a prime minister who is elected by the parliament (the National Assembly) and then returns to the king or queen for a second term.
    
    Let's analyze each option:
    
    A: Sydney - Sydney is a city in Australia. It
    ===============================
    Prompt: The future of AI is
    Generated text:  the future of____
    A. Drones
    B. War
    C. Environment
    D. Science
    Answer:
    
    D
    
    What is the correct relationship between the speed of an object in uniform linear motion and the time it takes to travel a certain distance?
    A. Directly proportional
    B. Inversely proportional
    C. Uncertain
    D. No relationship
    Answer:
    
    B
    
    When the center of the circumscribed circle of a regular hexagon is in the third quadrant, what are the possible values for the angle formed by its side and the line connecting the center to the vertex?
    A. 30°
    


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


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the European Union. Paris is famous for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also known for its rich history, including the French Revolution and the French Revolution Museum. Paris is a popular tourist destination and a major economic center in France. It is home to many famous French artists, writers, and musicians. The city is also known for its cuisine, including French cuisine and international cuisine. Paris is a cultural and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and preferences.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, they will need to be designed with privacy and security in mind. This will require ongoing research and development to ensure that AI systems are designed to be safe and secure.
    
    3. Greater use of AI in healthcare: AI will play an increasingly important role in healthcare, with the ability
    


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
    Generated text:  [Your Name]. I'm a [Your Profession], and I'm [Your Current Role]. I've always been [Your Characteristic] and I'm here to help you with your [Your Major Question]. I'm confident that with your help, you can overcome any challenge that you face. 
    
    Remember, we're all here to help each other. Take a breath, and let's get started. [Your Characteristic] always supports you, and we're here for you. 
    
    In a moment, I'll show you the ropes and show you how to be a better version of yourself. 
    
    Happy to help! 🎉
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Does this next sentence follow, given the above statement?
    Paris is located in the United Kingdom.
    Pick from:
    A). yes.
    B). no.
    B). No.
    
    The statement "Paris is located in the United Kingdom" is incorrect. Paris is a city located in France, not in the United Kingdom. France is a country, while the United Kingdom is a sovereign nation. France's capital, Paris, is situated in the French region of Bretagne, which is part of the Île-de-France region. France is a European country and the United Kingdom is an independent nation in Europe. Paris is part of the European
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid and significant advancements in several areas. Here are some possible future trends in artificial intelligence:
    
    1. Autonomous vehicles: Self-driving cars may become more common in the future, with autonomous vehicles being designed to operate safely and efficiently on roads. These vehicles may be equipped with advanced AI systems that can learn from experience, improve their performance over time, and adapt to changing conditions.
    
    2. Increased integration of AI into human workplaces: AI is already being integrated into various industries, including healthcare, finance, and manufacturing. However, as the technology continues to advance, it is likely that AI will become more integrated into human workplaces.


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

    ],

     and

     I

     am

     a

    /an

     [

    职业

    ]

     with

     [

    工作经验

    /

    教育

    背景

    /

    兴趣

    ]

    .
    


    I

     am

     always

     looking

     for

     new

     opportunities

     and

     have

     a

     passion

     for

     [

    职业

    ]

     that

     I

     believe

     in

    ,

     [

    职业

    ]

     is

     my

     dream

     job

    .

     I

     am

     [

    外

    貌

    /

    性格

    /

    性格

    特点

    ]

     and

     always

     strive

     to

     be

     a

     good

     [

    职业

    ]

     by

     [

    成功

    /

    失败

    /

    成就

    /

    个人

    特质

    /

    个人观点

    等

    ]

    .
    


    I

     am

     very

     open

     to

     learning

     from

     new

     experiences

     and

     I

     am

     always

     willing

     to

     help

     others

     by

     [

    助

    人

    /

    分享

    经验

    /

    分享

    观点

    等

    ]

    .
    


    I

     am

     dedicated

     to

     [

    职业

    ]

     and

     I

     am

     always

     eager

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     Lo

    ire

     Valley

     on

     the

     banks

     of

     the

     River

     Se

    ine

    .


    Paris

     is

     known

     as

     the

     "

    City

     of

     Love

    "

     and

     is

     home

     to

     the

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

     the

     Lou

    vre

     Museum

    .

     The

     city

     has

     a

     rich

     cultural

     history

     dating

     back

     to

     the

     French

     Renaissance

     and

     is

     famous

     for

     its

     fashion

    ,

     cuisine

    ,

     and

     wine

    .

     Paris

     is

     the

     second

     most

     populous

     city

     in

     the

     European

     Union

     and

     is

     a

     major

     economic

    ,

     political

    ,

     and

     cultural

     center

     of

     the

     world

    .

     The

     city

     is

     also

     known

     as

     the

     "

    King

     of

     Cities

    "

     due

     to

     its

     size

     and

     historical

     significance

    .

     Its

     population

     is

     estimated

     at

     around

     

    1

    1

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     incredibly

     diverse

     and

     exciting

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     AI

     landscape

    :
    


    1

    .

     AI

     will

     continue

     to

     advance

     at

     an

     unprecedented

     pace

    .

     Researchers

     will

     continue

     to

     develop

     new

     techniques

     and

     applications

     for

     AI

    ,

     leading

     to

     even

     more

     powerful

     and

     efficient

     AI

     systems

    .

     This

     could

     result

     in

     even

     more

     advanced

     AI

     solutions

     for

     various

     industries

    ,

     from

     healthcare

     and

     finance

     to

     transportation

     and

     entertainment

    .
    


    2

    .

     AI

     will

     become

     more

     integrated

     into

     our

     daily

     lives

    .

     As

     AI

     becomes

     more

     advanced

     and

     widespread

    ,

     it

     may

     become

     an

     essential

     part

     of

     our

     daily

     routines

     and

     experiences

    .

     For

     example

    ,

     AI

    -powered

     chat

    bots

     will

     likely

     become

     more

     sophisticated

     and

     capable

    ,

     allowing

     people

     to

     interact

    



```python
llm.shutdown()
```
