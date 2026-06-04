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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]

    Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:07,  5.92it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:07,  5.92it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:07,  5.92it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:07,  5.92it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:07,  5.92it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:07,  5.92it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:04, 10.01it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:04, 10.01it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:04, 10.01it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:04, 10.01it/s]

    Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:04, 10.01it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 13.54it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 21.31it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 21.31it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 21.31it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 21.31it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 21.31it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 21.31it/s]

    Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 21.31it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:01, 21.31it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 37.18it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 37.18it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 37.18it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 37.18it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 37.18it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 37.18it/s] 

    Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 37.18it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 37.18it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 44.21it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 52.31it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 52.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=38.29 GB):   2%|▏         | 1/58 [00:00<00:12,  4.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.26 GB):   2%|▏         | 1/58 [00:00<00:12,  4.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.26 GB):   3%|▎         | 2/58 [00:00<00:09,  5.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.25 GB):   3%|▎         | 2/58 [00:00<00:09,  5.85it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=38.25 GB):   5%|▌         | 3/58 [00:00<00:07,  7.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.25 GB):   5%|▌         | 3/58 [00:00<00:07,  7.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.25 GB):   5%|▌         | 3/58 [00:00<00:07,  7.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.25 GB):   9%|▊         | 5/58 [00:00<00:05,  9.43it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.24 GB):   9%|▊         | 5/58 [00:00<00:05,  9.43it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=38.23 GB):   9%|▊         | 5/58 [00:00<00:05,  9.43it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.23 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.85it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.23 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.23 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.23 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.23 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.50it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=38.22 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.22 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.22 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.22 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.22 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.22 GB):  24%|██▍       | 14/58 [00:01<00:02, 16.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.21 GB):  24%|██▍       | 14/58 [00:01<00:02, 16.95it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=38.21 GB):  24%|██▍       | 14/58 [00:01<00:02, 16.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.20 GB):  24%|██▍       | 14/58 [00:01<00:02, 16.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.20 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.20 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.20 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.20 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.20 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.18 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.09it/s]

    Capturing num tokens (num_tokens=960 avail_mem=38.19 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.09it/s] Capturing num tokens (num_tokens=896 avail_mem=38.19 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.09it/s]Capturing num tokens (num_tokens=832 avail_mem=38.18 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.09it/s]Capturing num tokens (num_tokens=832 avail_mem=38.18 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.08it/s]Capturing num tokens (num_tokens=768 avail_mem=38.18 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.08it/s]Capturing num tokens (num_tokens=704 avail_mem=38.18 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.08it/s]Capturing num tokens (num_tokens=640 avail_mem=38.17 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.08it/s]Capturing num tokens (num_tokens=576 avail_mem=38.17 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.08it/s]

    Capturing num tokens (num_tokens=576 avail_mem=38.17 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.52it/s]Capturing num tokens (num_tokens=512 avail_mem=38.16 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.52it/s]Capturing num tokens (num_tokens=480 avail_mem=38.17 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.52it/s]Capturing num tokens (num_tokens=448 avail_mem=38.17 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.52it/s]Capturing num tokens (num_tokens=416 avail_mem=38.17 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.52it/s]Capturing num tokens (num_tokens=416 avail_mem=38.17 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=384 avail_mem=38.17 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=352 avail_mem=38.16 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=320 avail_mem=38.16 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.72it/s]

    Capturing num tokens (num_tokens=288 avail_mem=38.15 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=288 avail_mem=38.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.92it/s]Capturing num tokens (num_tokens=256 avail_mem=38.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.92it/s]Capturing num tokens (num_tokens=240 avail_mem=38.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.92it/s]Capturing num tokens (num_tokens=224 avail_mem=38.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.92it/s]Capturing num tokens (num_tokens=208 avail_mem=38.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.92it/s]Capturing num tokens (num_tokens=208 avail_mem=38.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=192 avail_mem=38.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=176 avail_mem=38.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.09it/s]

    Capturing num tokens (num_tokens=160 avail_mem=38.14 GB):  69%|██████▉   | 40/58 [00:02<00:00, 32.09it/s]Capturing num tokens (num_tokens=144 avail_mem=38.13 GB):  69%|██████▉   | 40/58 [00:02<00:00, 32.09it/s]Capturing num tokens (num_tokens=144 avail_mem=38.13 GB):  76%|███████▌  | 44/58 [00:02<00:00, 32.71it/s]Capturing num tokens (num_tokens=128 avail_mem=38.13 GB):  76%|███████▌  | 44/58 [00:02<00:00, 32.71it/s]Capturing num tokens (num_tokens=112 avail_mem=38.13 GB):  76%|███████▌  | 44/58 [00:02<00:00, 32.71it/s]Capturing num tokens (num_tokens=96 avail_mem=38.12 GB):  76%|███████▌  | 44/58 [00:02<00:00, 32.71it/s] Capturing num tokens (num_tokens=80 avail_mem=38.12 GB):  76%|███████▌  | 44/58 [00:02<00:00, 32.71it/s]Capturing num tokens (num_tokens=80 avail_mem=38.12 GB):  83%|████████▎ | 48/58 [00:02<00:00, 33.30it/s]Capturing num tokens (num_tokens=64 avail_mem=38.12 GB):  83%|████████▎ | 48/58 [00:02<00:00, 33.30it/s]

    Capturing num tokens (num_tokens=48 avail_mem=38.11 GB):  83%|████████▎ | 48/58 [00:02<00:00, 33.30it/s]Capturing num tokens (num_tokens=32 avail_mem=38.11 GB):  83%|████████▎ | 48/58 [00:02<00:00, 33.30it/s]Capturing num tokens (num_tokens=28 avail_mem=38.10 GB):  83%|████████▎ | 48/58 [00:02<00:00, 33.30it/s]Capturing num tokens (num_tokens=28 avail_mem=38.10 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.72it/s]Capturing num tokens (num_tokens=24 avail_mem=38.10 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.72it/s]Capturing num tokens (num_tokens=20 avail_mem=38.10 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.72it/s]Capturing num tokens (num_tokens=16 avail_mem=38.10 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.72it/s]Capturing num tokens (num_tokens=12 avail_mem=38.09 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.72it/s]Capturing num tokens (num_tokens=12 avail_mem=38.09 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.28it/s]Capturing num tokens (num_tokens=8 avail_mem=38.09 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.28it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=38.09 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.28it/s]Capturing num tokens (num_tokens=4 avail_mem=38.09 GB): 100%|██████████| 58/58 [00:02<00:00, 23.56it/s]


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
    Generated text:  Tracy White and I am a professional coach for GirlsDevelopers. I have worked with thousands of women to grow their skills in all areas of sports. I am also a teacher and I have helped many women and girls move through different age levels and stages of life. If you are interested in working with me to create a customized program for your girls and girls from your community, contact me. As you can see, my life story is a story of perseverance, hard work and a desire to help women and girls reach their potential. I hope that you will find this info valuable. I would love to hear what you think about my story.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a civilian who is not a member of the military, has a formal education degree, and is chosen for their expertise and leadership qualities.
    
    Is the following statement true?
    "Chester A. Ford served as the president of the United States."
    
    Let me think through this like we're playing a game of deduction, okay?
    
    To determine if the statement "Chester A. Ford served as the president of the United States" is true, let's break down the key points:
    
    1. **Purpose**: The statement is about the presidency, specifically for a civilian with a formal education degree.
    2. **Expertise and Leadership**: The president is
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. Brussels C. Berlin D. Moscow
    Answer: A
    
    Which of the following statements about the legal framework of the European Union is correct?
    A. The European Union is not a legal entity, it is merely a legal relationship body.
    B. The EU's legal framework is composed of several basic treaties, institutional conventions, and other legal norms.
    C. The EU's legal framework includes both domestic legislation and international law.
    D. The legal framework of the EU is composed of two legal departments: the European Union legal framework and the EU legislative framework.
    Answer: B
    
    Which of the following statements is true
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable, but we know that the next big AI advance will be something that could be seen as revolutionary. However, we cannot say that this is the most advanced machine learning tool ever developed. In this post, we will discuss the most advanced machine learning tools and give an overview of the machine learning algorithms they use.
    
    Machine Learning Algorithms: The Basics
    
    Machine learning algorithms are a class of algorithms used to develop algorithms that can learn from experience and improve with it. This is the starting point of machine learning and is a way of learning without being explicitly programmed. This is a prerequisite for many machine learning applications. Machine learning algorithms can be supervised learning


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character or profession]. I enjoy [insert a short description of your interests or hobbies]. I'm always looking for new experiences and learning new things. What's your favorite hobby or activity? I'm always looking for new challenges and opportunities to grow and learn. What's your favorite book or movie? I'm always looking for new perspectives and ideas to explore. What's your favorite place to go? I'm always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in the European Union and the third-largest city in the world by population. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major center for business, finance, and tourism. Paris is a city that is constantly evolving and changing, with new developments and cultural events taking place throughout the year. The city is also known for its cuisine, with many famous French dishes such as croissants, boudin, and escargot.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread adoption in healthcare.
    
    2. AI in finance: AI is already being used in finance to improve fraud detection, risk management, and trading algorithms. As AI technology continues to improve, we can expect to see even more widespread adoption in finance.
    
    3. AI in manufacturing: AI is already
    


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
    Generated text:  [Name], and I'm a [field of work], a [number] year old [profession]. I have a [field of work] degree and have been in the industry for [number] years. I have been passionate about [field of work] since I was [number] old and have been working hard to [field of work] ever since. I'm always looking for ways to [field of work] and always have a new idea or skill to add to my portfolio. I'm a [field of work] enthusiast and always strive to improve and be better. I'm a [field of work] enthusiast and always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and has a rich history and culture. It is known for its beautiful architecture, iconic landmarks, and lively nightlife. The city is home to many world-renowned museums, parks, and landmarks, including the Eiffel Tower and Notre-Dame Cathedral. Paris is also an important economic and cultural hub, attracting visitors and residents from around the world. Despite its size, Paris is a peaceful and welcoming city that offers people an escape from the city's hustle and bustle. The French government has worked hard to preserve and protect the city's unique heritage and culture, which is evident in the city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating and is likely to evolve rapidly in the coming years. Here are some possible trends:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in various fields, it is likely to attract more ethical concerns and questions. Developers and researchers will need to consider issues such as fairness, transparency, and accountability when designing AI systems.
    
    2. Increased use of AI in healthcare: AI is already being used in healthcare, from diagnosing diseases to predicting patient outcomes. As AI becomes more advanced, it is likely to be used in more clinical settings, including personalized treatment plans for patients.
    
    3. Increased use of AI in manufacturing: AI is


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

    'm

     an

     AI

     language

     model

    .

     I

    'm

     here

     to

     assist

     you

     with

     any

     questions

     or

     tasks

     you

     may

     have

    ,

     and

     I

    'm

     here

     to

     provide

     information

     and

     guidance

     on

     a

     wide

     range

     of

     topics

    .

     Whether

     you

     need

     help

     with

     writing

    ,

     research

    ,

     or

     anything

     else

    ,

     I

    'm

     here

     to

     help

    .

     What

     can

     I

     do

     for

     you

     today

    ?

     Let

     me

     know

     and

     I

    'll

     do

     my

     best

     to

     assist

     you

    .

     [

    Insert

     name

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     largest

     urban

     area

     in

     the

     world

     by

     population

    .

     It

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

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

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     and

     its

     rich

     culture

    ,

     history

    ,

     and

     cuisine

     make

     it

     a

     major

     center

     of

     the

     French

     Republic

    .

     The

     city

     is

     home

     to

     many

     notable

     historical

     and

     cultural

     sites

    ,

     including

     the

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     It

     is

     also

     a

     major

     cultural

     and

     commercial

     hub

    ,

     with

     many

     museums

    ,

     theaters

    ,

     and

     restaurants

    .

     The

     city

     is

     a

     melting

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     exponential

     growth

    ,

     innovation

    ,

     and

     the

     integration

     of

     more

     advanced

     technologies

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     could

     be

     expected

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     sophistication

    :

     As

     AI

     technology

     advances

    ,

     we

     are

     likely

     to

     see

     more

     sophisticated

     algorithms

     and

     models

     that

     can

     process

     large

     amounts

     of

     data

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

     increasing

     use

     of

     AI

     in

     transportation

    ,

     we

     can

     expect

     to

     see

     more

     autonomous

     vehicles

     in

     the

     future

    .

     These

     vehicles

     will

     be

     equipped

     with

     advanced

     sensors

    ,

     cameras

    ,

     and

     AI

     algorithms

     that

     will

     help

     them

     to

     navigate

     the

     roads

     safely

     and

     efficiently

    .
    


    3

    .

     Medical

     applications

    :

     AI

     will

     play

     a

    



```python
llm.shutdown()
```
