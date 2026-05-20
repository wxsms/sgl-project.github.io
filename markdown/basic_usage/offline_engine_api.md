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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.05it/s]


    2026-05-20 11:47:16,853 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 11:47:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:41,  4.93s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.90it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.90it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.90it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.90it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.90it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.90it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.90it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.90it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.90it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 18.62it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 18.62it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 18.62it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 18.62it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 18.62it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 18.62it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.62it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 18.62it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 18.62it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 18.62it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 26.44it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 26.44it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 26.44it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 26.44it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 26.44it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 26.44it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 26.44it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 26.44it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 26.44it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 33.54it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 33.54it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 33.54it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 33.54it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 33.54it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 33.54it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.37 GB):   2%|▏         | 1/58 [00:00<00:07,  7.77it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   2%|▏         | 1/58 [00:00<00:07,  7.77it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:06,  8.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:06,  8.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:06,  8.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.34 GB):   7%|▋         | 4/58 [00:00<00:04, 11.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   7%|▋         | 4/58 [00:00<00:04, 11.06it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):   7%|▋         | 4/58 [00:00<00:04, 11.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):  10%|█         | 6/58 [00:00<00:03, 13.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):  10%|█         | 6/58 [00:00<00:03, 13.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):  10%|█         | 6/58 [00:00<00:03, 13.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.35it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=72.31 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.35it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.31 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.30 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.30 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.28it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.28it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.29 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  24%|██▍       | 14/58 [00:00<00:02, 18.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  29%|██▉       | 17/58 [00:01<00:02, 20.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.28 GB):  29%|██▉       | 17/58 [00:01<00:02, 20.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  29%|██▉       | 17/58 [00:01<00:02, 20.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  29%|██▉       | 17/58 [00:01<00:02, 20.23it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  29%|██▉       | 17/58 [00:01<00:02, 20.23it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.78it/s]Capturing num tokens (num_tokens=960 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.78it/s] Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.78it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.78it/s]Capturing num tokens (num_tokens=768 avail_mem=72.26 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.78it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.78it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.12it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.12it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.12it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.12it/s]

    Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.12it/s]Capturing num tokens (num_tokens=448 avail_mem=72.25 GB):  45%|████▍     | 26/58 [00:01<00:01, 30.12it/s]Capturing num tokens (num_tokens=448 avail_mem=72.25 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.67it/s]Capturing num tokens (num_tokens=416 avail_mem=72.25 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.67it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.67it/s]Capturing num tokens (num_tokens=352 avail_mem=72.24 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.67it/s]Capturing num tokens (num_tokens=320 avail_mem=72.21 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.67it/s]Capturing num tokens (num_tokens=320 avail_mem=72.21 GB):  60%|██████    | 35/58 [00:01<00:00, 31.58it/s]Capturing num tokens (num_tokens=288 avail_mem=71.95 GB):  60%|██████    | 35/58 [00:01<00:00, 31.58it/s]

    Capturing num tokens (num_tokens=256 avail_mem=72.17 GB):  60%|██████    | 35/58 [00:01<00:00, 31.58it/s]Capturing num tokens (num_tokens=240 avail_mem=72.20 GB):  60%|██████    | 35/58 [00:01<00:00, 31.58it/s]Capturing num tokens (num_tokens=224 avail_mem=72.19 GB):  60%|██████    | 35/58 [00:01<00:00, 31.58it/s]Capturing num tokens (num_tokens=224 avail_mem=72.19 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.87it/s]Capturing num tokens (num_tokens=208 avail_mem=72.18 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.87it/s]Capturing num tokens (num_tokens=192 avail_mem=72.17 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.87it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.17 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.87it/s]Capturing num tokens (num_tokens=176 avail_mem=72.17 GB):  72%|███████▏  | 42/58 [00:01<00:00, 26.93it/s]Capturing num tokens (num_tokens=160 avail_mem=72.16 GB):  72%|███████▏  | 42/58 [00:01<00:00, 26.93it/s]Capturing num tokens (num_tokens=144 avail_mem=72.15 GB):  72%|███████▏  | 42/58 [00:01<00:00, 26.93it/s]Capturing num tokens (num_tokens=128 avail_mem=72.15 GB):  72%|███████▏  | 42/58 [00:01<00:00, 26.93it/s]Capturing num tokens (num_tokens=128 avail_mem=72.15 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.19it/s]Capturing num tokens (num_tokens=112 avail_mem=72.14 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.19it/s]Capturing num tokens (num_tokens=96 avail_mem=72.14 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.19it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=72.13 GB):  78%|███████▊  | 45/58 [00:02<00:00, 27.19it/s]Capturing num tokens (num_tokens=80 avail_mem=72.13 GB):  83%|████████▎ | 48/58 [00:02<00:00, 27.69it/s]Capturing num tokens (num_tokens=64 avail_mem=72.12 GB):  83%|████████▎ | 48/58 [00:02<00:00, 27.69it/s]Capturing num tokens (num_tokens=48 avail_mem=72.05 GB):  83%|████████▎ | 48/58 [00:02<00:00, 27.69it/s]Capturing num tokens (num_tokens=32 avail_mem=72.11 GB):  83%|████████▎ | 48/58 [00:02<00:00, 27.69it/s]Capturing num tokens (num_tokens=32 avail_mem=72.11 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.04it/s]Capturing num tokens (num_tokens=28 avail_mem=72.10 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.04it/s]Capturing num tokens (num_tokens=24 avail_mem=72.05 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.04it/s]Capturing num tokens (num_tokens=20 avail_mem=72.09 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.04it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.09 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.56it/s]Capturing num tokens (num_tokens=16 avail_mem=72.07 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.56it/s]Capturing num tokens (num_tokens=12 avail_mem=71.80 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.56it/s]Capturing num tokens (num_tokens=8 avail_mem=71.79 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.56it/s] Capturing num tokens (num_tokens=8 avail_mem=71.79 GB):  98%|█████████▊| 57/58 [00:02<00:00, 27.99it/s]Capturing num tokens (num_tokens=4 avail_mem=71.78 GB):  98%|█████████▊| 57/58 [00:02<00:00, 27.99it/s]Capturing num tokens (num_tokens=4 avail_mem=71.78 GB): 100%|██████████| 58/58 [00:02<00:00, 24.33it/s]


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
    Generated text:  Alex. I am a middle school student. I have a hobby—photography. I like to take beautiful pictures with my camera. Sometimes, I even go out and take pictures of my friends. Now, I want to share some of my favorite pictures with you. Here is one of my favorite pictures: This is a photo of my family. My parents are in the middle, and my sister and I are standing on the left. My brother is on the right. My little brother is playing with a toy car. My mother is smiling. She is holding a picture of her favorite artist. The photo is taken in front of the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very busy man. He has to deal with many important things every day. He has to work hard to make sure that the country is safe and the people have the food they eat. He has to make sure that people are educated so that they can be productive members of the society. He also has to deal with all kinds of problems that may affect the country. He has to be ready to do whatever is necessary to make the country safe. He is always ready to listen to people's ideas. He is always ready to help the people when they need it. He listens to people's ideas and helps them when they are in need
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It was originally an ancient city that was established by the French kings. There have been many rulers in the city. Today, it is known as the cultural capital of France. Many French people travel to Paris for holidays and cultural events. There is a famous monument, the Eiffel Tower, in Paris.
    
    The French Revolution had a major impact on Paris. It was during this time that Louis XVI was imprisoned by the royalists and executed by guillotine. In the following years, many other rulers of France were also executed by the royalists. It was during this time that many of the old buildings and streets of Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  very uncertain. The questions for today are:
    
    1. Will AI change the world?
    2. How will AI be used?
    3. How will AI be developed?
    4. How will the market for AI be generated?
    5. How will AI impact the economy?
    6. How will AI impact human life?
    7. How will AI be used in education?
    8. How will AI be used in healthcare?
    
    Make sure to provide examples, suggestions, and explanations for each question. Provide a summary of the questions to the best of your ability, and summarize the importance of understanding the potential impact of AI on society in your response. Make sure


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you. What can you tell me about yourself? I'm a [job title] at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is the capital of France and the largest city in the European Union. It is also the birthplace of many famous French artists, writers, and composers. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of people, with a diverse population of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, AI is likely to play an increasingly important role in shaping the future of work, as it is expected to automate many of the tasks that are currently performed by humans. Finally, AI is likely to continue to be used for good, with applications such as climate change mitigation, public health, and disaster response being areas where AI is expected to have a significant impact. Overall,
    


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
    Generated text:  Emily. I'm a cheerful, optimistic person who loves spending time with friends and exploring new places. I'm always eager to learn new things and have a love for adventure. I'm always looking for the next great discovery and have a good sense of humor. I'm here to help anyone who needs a little cheer in their day. Let's make some new memories together! What's your favorite hobby or activity to do? As a language model, I don't have the ability to have hobbies or activities, but I can suggest a few fun activities to try! What's your favorite hobby or activity to do? As a language model,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It's also a bustling hub for art, fashion, and culture. 
    
    The city is renowned for its rich history and beautiful architecture, including the Palace of Versailles, the Tuileries Gardens, and the Moulin Rouge. The city also hosts numerous events throughout the year, including the World Cup, the Glastonbury Festival, and the Eurovision Song Contest.
    
    The weather in Paris can vary greatly from warm summer months to cold winter days, with the city known for its coffee culture and its diverse array
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by significant progress, innovation, and growth. Here are some possible trends we can expect to see in the field of AI in the coming years:
    
    1. Increased Use of AI in Healthcare: The use of AI in healthcare is likely to grow as more healthcare providers adopt AI-based tools to improve patient outcomes, reduce costs, and improve patient experience. AI can be used to analyze medical images, diagnose diseases, and predict patient outcomes. The development of new AI-powered tools and platforms could make healthcare more accessible and efficient.
    
    2. Integration of AI in Customer Service: The integration of AI in customer service can help businesses to better


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

     

    2

    2

    -year

    -old

     [

    age

    ]

     student

     major

    ing

     in

     [

    major

    ].

     I

    've

     always

     been

     passionate

     about

     [

    profession

    ],

     and

     I

    've

     always

     wanted

     to

     [

    specific

     goal

     or

     achievement

    ].

     I

    'm

     currently

     [

    occupation

    ],

     and

     I

    'm

     always

     on

     the

     lookout

     for

     opportunities

     to

     [

    exc

    use

     for

     action

    ].

     If

     you

     have

     any

     questions

     or

     would

     like

     to

     learn

     more

     about

     me

    ,

     please

     feel

     free

     to

     ask

    .

     [

    Your

     name

    ]

     can

    't

     promise

     anything

     in

     return

    ,

     but

     I

     believe

     in

     being

     true

     to

     myself

     and

     pursuing

     what

     makes

     me

     happy

    .

     I

    'm

     confident

     that

     I

    'll

     make

     a

     great

     fit

     for

     the

     [

    company

    ]

     team

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     often

     referred

     to

     as

     the

     “

    City

     of

     Love

    ”

     due

     to

     its

     iconic

     Notre

    -D

    ame

     Cathedral

    ,

     which

     is

     one

     of

     the

     most

     recognizable

     symbols

     of

     the

     city

    .


    Paris

     has

     a

     long

     and

     stor

    ied

     history

     dating

     back

     to

     the

     

    6

    th

     century

     and

     has

     undergone

     numerous

     changes

     throughout

     the

     years

    ,

     including

     the

     Nap

    ole

    onic

     Wars

    ,

     the

     French

     Revolution

    ,

     and

     the

     French

     Second

     Republic

    .


    Paris

     is

     home

     to

     many

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

    '

    Or

    say

    ,

     as

     well

     as

     theaters

    ,

     art

     galleries

    ,

     and

     restaurants

    .

     The

     city

     is

     also

     known

     for

     its

     wine

     production

    ,

     particularly

     in

     the

     V

    os

    ges

     region

    ,

     and

     its

     cultural

     offerings

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     constantly

     evolving

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

     future

     of

     AI

    :
    


    1

    .

     Improved

     natural

     language

     processing

    :

     With

     the

     help

     of

     advances

     in

     machine

     learning

    ,

     we

     can

     expect

     to

     see

     more

     accurate

     and

     effective

     natural

     language

     processing

     in

     the

     future

    .

     This

     could

     mean

     more

     efficient

     speech

     recognition

     and

     understanding

     of

     human

     language

    ,

     as

     well

     as

     more

     accurate

     language

     translation

    .
    


    2

    .

     Enhanced

     robotics

     and

     automation

    :

     With

     the

     development

     of

     more

     advanced

     AI

     algorithms

    ,

     we

     could

     see

     more

     sophisticated

     robotics

     that

     can

     perform

     tasks

     that

     are

     currently

     out

     of

     the

     scope

     of

     human

     ability

    ,

     such

     as

     sim

    ulating

     human

     emotions

    ,

     managing

     complex

     social

     interactions

    ,

     or

     even

     performing

     complex

     tasks

     that

     require

     human

    



```python
llm.shutdown()
```
