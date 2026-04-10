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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.10it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.07it/s]


    2026-04-10 00:55:40,211 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 00:55:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:39,  2.79s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  5.97it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.03it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.03it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.03it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.03it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.03it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.03it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.03it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.03it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.28it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 33.98it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 33.98it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 33.98it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 33.98it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 33.98it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 33.98it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 33.98it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 37.78it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 37.78it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 37.78it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 37.78it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 37.78it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 37.78it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 37.78it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 37.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:05, 10.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:05, 10.45it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.76 GB):   3%|▎         | 2/58 [00:00<00:05, 10.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.76 GB):   7%|▋         | 4/58 [00:00<00:04, 11.41it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   7%|▋         | 4/58 [00:00<00:04, 11.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   7%|▋         | 4/58 [00:00<00:04, 11.41it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:04, 12.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:04, 12.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:04, 12.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.75 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.82it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.82it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.83it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.19it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.72 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.56it/s]Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.56it/s] Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.56it/s]

    Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.56it/s]Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.79it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.79it/s]Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.79it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.79it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.79it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.74it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.74it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.74it/s]

    Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.74it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.74it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.22it/s]Capturing num tokens (num_tokens=384 avail_mem=118.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.22it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.22it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.22it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.22it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.86it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.86it/s]

    Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.86it/s]Capturing num tokens (num_tokens=224 avail_mem=118.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.86it/s]Capturing num tokens (num_tokens=224 avail_mem=118.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.71it/s]Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.71it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.71it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.71it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.71it/s]

    Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 28.69it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 28.69it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 28.69it/s]Capturing num tokens (num_tokens=112 avail_mem=118.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 28.69it/s]Capturing num tokens (num_tokens=112 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.65it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.65it/s] Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.65it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:02<00:00, 28.65it/s]

    Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:02<00:00, 29.01it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:02<00:00, 29.01it/s]Capturing num tokens (num_tokens=32 avail_mem=118.63 GB):  84%|████████▍ | 49/58 [00:02<00:00, 29.01it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  84%|████████▍ | 49/58 [00:02<00:00, 29.01it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.95it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.95it/s]Capturing num tokens (num_tokens=20 avail_mem=118.62 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.95it/s]Capturing num tokens (num_tokens=16 avail_mem=118.62 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.95it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  90%|████████▉ | 52/58 [00:02<00:00, 28.95it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  97%|█████████▋| 56/58 [00:02<00:00, 30.21it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  97%|█████████▋| 56/58 [00:02<00:00, 30.21it/s] Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  97%|█████████▋| 56/58 [00:02<00:00, 30.21it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 24.75it/s]


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
    Generated text:  Seryat and I’m the founder of the Innate Learning Group.
    I am passionate about working with children and adults on developing skills and increasing confidence. At the Innate Learning Group, I offer a range of coaching and mentoring services, including personal coaching, coaching and mentoring, and curriculum development.
    In my previous role as an instructor, I taught and coached the children of members of the Royal Family. As an instructor, I knew that I had to create a unique and authentic experience for the children, but I also had to ensure that they learned something new. I used my personal experience as a family member and the love and support I
    ===============================
    Prompt: The president of the United States is
    Generated text:  a well-known figure. He has a speech that he often gives to the country and a speech given by him to the general public. While giving his speech to the country, the president is more likely to be involved in the implementation of the policies. In contrast, the president who gives his speech to the general public is more likely to be involved in the implementation of the policies. Is the following statement true or false? To determine if the given statement is true or false, we need to understand the relationship between the president's speech and the implementation of policies.
    
    1. The president's speech to the country is typically a formal and detailed public
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A) Paris
    B) Lyon
    C) London
    D) Rome
    The capital of France is Paris. 
    
    To break it down:
    
    - Paris, France, is located in the south of France, bordering the Mediterranean Sea to the north, the Atlantic Ocean to the west, and the English Channel to the east.
    - It is the largest city in France and is the seat of the French government and the second most populous city in the European Union after Rome.
    - The city is named after the famous French poet and writer Victor Hugo, who lived there from 1830 to 1885.
    -
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to involve more sophisticated and complex models that can understand and reason about the world beyond the boundaries of their programming. This requires significant advancements in the field of machine learning and the development of new algorithms and techniques for training deep neural networks. Additionally, there will be a greater focus on developing AI technologies that are more ethical and accountable, as well as providing more transparency and explainability for AI systems.
    To achieve these goals, researchers and developers will need to collaborate and work together to create new and innovative techniques for training AI models, as well as new tools and frameworks that can help with the development of ethical and accountable AI technologies. It will also


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. It is also the birthplace of the French Revolution and the current capital of France. Paris is a bustling metropolis with a rich history and a vibrant culture that attracts millions of visitors annually. Its status as the world's most populous city is due to its population of over 20 million people. The city is also known for its fashion industry, art scene, and food culture. Paris is a city that has played a significant role in shaping French identity and culture. Its status as the capital of France is recognized globally, and it
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient solutions to complex problems.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more transparent and accountable AI systems, as well as more robust safeguards against bias and discrimination.
    
    3. Increased
    


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
    Generated text:  [Your Name]. I’m a [Your Field or Job Title] with [Your Company/Institution] and I have been working in this field for [Number of Years] years. I am [Your Hobbies, Interests, or Passion] and I enjoy [Your Personal Tagline].
    My [Your Job Title] has been a part of [Your Company/Institution] for [Number of Years]. My career has been dedicated to [Your Professional Purpose or Mission]. I’ve had the privilege of traveling the world, meeting people from all walks of life, and having the opportunity to influence and inspire others. I believe that
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its magnificent architecture, vibrant cultural scene, and international reputation. Paris is one of the world’s most visited cities and hosts a multitude of festivals and events throughout the year. It is also home to numerous museums, galleries, and art galleries, including the Louvre. Overall, Paris is a city that is synonymous with elegance, romance, and cultural richness. Its status as the world's most populous city has further cemented its significance in French and global society. Can you provide more information on the iconic landmarks and cultural events that take place in Paris? Yes, here are some additional details on the iconic landmarks and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by significant advancements in several areas, driven by the integration of more sophisticated algorithms, increased reliance on data, and the development of new hardware. Here are some potential future trends in AI:
    
    1. Increased Robustness: AI will become even more resilient and capable of adapting to complex and unpredictable environments. It will be able to learn from errors and failures in its actions, while also continuously improving its performance over time.
    
    2. Personalization and Augmentation: AI will be able to provide highly personalized and augmented experiences for users, while also augmenting users with more information and context about the world around them.
    
    3. Improved


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

     John

     and

     I

    'm

     a

     [

    insert

     your

     profession

    ]

     at

     [

    insert

     your

     company

     name

    ].

     I

    've

     always

     been

     passionate

     about

     [

    insert

     a

     project

     or

     area

     of

     interest

     that

     you

     enjoy

     and

     that

     you

    've

     worked

     on

     in

     the

     past

    ],

     and

     I

    'm

     always

     looking

     to

     learn

     and

     grow

     in

     my

     career

    .

     I

    'm

     confident

     that

     I

     can

     make

     a

     great

     contribution

     to

     the

     team

     at

     [

    insert

     your

     company

     name

    ],

     and

     I

    'm

     eager

     to

     apply

     my

     skills

     and

     expertise

     to

     our

     goals

     and

     objectives

    .

     Thank

     you

    .

     Write

     a

     short

    ,

     neutral

     self

    -int

    roduction

     for

     a

     fictional

     character

    .

     Hello

    ,

     my

     name

     is

     John

     and

     I

    'm

     a

     [

    insert

     your

     profession

    ]

     at

     [

    insert

     your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     encaps

    ulates

     the

     core

     information

     required

     to

     describe

     the

     capital

     city

     of

     France

    ,

     including

     its

     capital

     city

     name

    ,

     the

     country

     it

     belongs

     to

    ,

     and

     the

     country it

     is

     located

     in

    .

     The

     fact

     is

     brief

     and

     precise

    ,

     giving

     a

     clear

     and

     concise

     understanding

     of

     the

     subject

    .

     It

     does

     not

     include

     any

     additional details

     or

     assumptions

     beyond

     the

     essential

     information

     needed

     for

     comprehension

    .

     This

     statement

     serves

     as a

     strong

     basis

     for

     an

     informative

     response

     without

     needing

     to

     include

     any

     outside

     information

     or

     assumptions

    .

     


    Additional

     context

     about

     Paris

    :


    -

     Located

     in

     the

     Î

    le

    -de

    -F

    rance

     region

     of

     France

    ,

     it

     is

     the

     largest

     city

     in

     the

     country

     and

     the

     most

     populous

    .


    -

     The

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

    ,

     but

     several

     trends

     are

     likely

     to

     shape

     it

     in

     the

     coming

     years

    :
    


    1.

     Increased

     use

     of

     AI

     in

     the

     healthcare

     industry

    :

     AI

     is

     becoming

     more widely

     used

     in

     the

     healthcare

     industry

    ,

     from

     improving

     diagnoses

     to

     predicting

     patient

     outcomes

    .

     This

     trend

     is

     expected

     to

     continue

     as

     AI

     technology

     improves

     and

     becomes

     more

     accessible

     to

     patients

    .
    


    2

    .

     Automation

     of

     customer

     service:

     AI

    -powered

     chat

    bots

     and

     virtual

     assistants

     are

     expected

     to

     become

     more

     common

     in

     customer

     service

    ,

     reducing

     the

     need

     for

     human

     agents

     and

     freeing

     up

     time

     for

     employees

     to

     focus

     on

     more

     complex

     tasks

    .
    


    3

    .

     Personal

    ization

     of

     services

    :

     AI

     will

     be

     used

     to

     provide

     personalized

     experiences

     to

     customers

    ,

     based

     on

     their

     preferences

     and

    



```python
llm.shutdown()
```
