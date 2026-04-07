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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.97it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.95it/s]


    2026-04-07 01:25:03,611 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 01:25:03] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.69it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.69it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.69it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.99it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.99it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.99it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.99it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.99it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.99it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.99it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.99it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.83it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.83it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.83it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.83it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.83it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.83it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.83it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.92it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.92it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.92it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.92it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.92it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.92it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.92it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.40it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.40it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.40it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.40it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.40it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.40it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.40it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.99 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.99 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.99 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.99 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.99 GB):   9%|▊         | 5/58 [00:00<00:02, 21.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.98 GB):   9%|▊         | 5/58 [00:00<00:02, 21.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.98 GB):   9%|▊         | 5/58 [00:00<00:02, 21.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.98 GB):   9%|▊         | 5/58 [00:00<00:02, 21.25it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=118.98 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.98 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.98 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.97 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.97 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.97 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.17it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=118.97 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.17it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.97 GB):  22%|██▏       | 13/58 [00:00<00:02, 15.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.97 GB):  22%|██▏       | 13/58 [00:00<00:02, 15.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.96 GB):  22%|██▏       | 13/58 [00:00<00:02, 15.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.96 GB):  22%|██▏       | 13/58 [00:00<00:02, 15.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.96 GB):  28%|██▊       | 16/58 [00:00<00:02, 17.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.95 GB):  28%|██▊       | 16/58 [00:00<00:02, 17.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.92 GB):  28%|██▊       | 16/58 [00:00<00:02, 17.95it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=118.91 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.91 GB):  33%|███▎      | 19/58 [00:01<00:02, 18.08it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.91 GB):  33%|███▎      | 19/58 [00:01<00:02, 18.08it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=118.89 GB):  33%|███▎      | 19/58 [00:01<00:02, 18.08it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.89 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.91it/s]Capturing num tokens (num_tokens=960 avail_mem=118.90 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.91it/s] Capturing num tokens (num_tokens=896 avail_mem=118.90 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.91it/s]Capturing num tokens (num_tokens=832 avail_mem=118.89 GB):  36%|███▌      | 21/58 [00:01<00:02, 14.91it/s]Capturing num tokens (num_tokens=832 avail_mem=118.89 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.61it/s]Capturing num tokens (num_tokens=768 avail_mem=118.89 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.61it/s]Capturing num tokens (num_tokens=704 avail_mem=118.89 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.61it/s]

    Capturing num tokens (num_tokens=640 avail_mem=118.89 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.61it/s]Capturing num tokens (num_tokens=576 avail_mem=118.89 GB):  41%|████▏     | 24/58 [00:01<00:01, 17.61it/s]Capturing num tokens (num_tokens=576 avail_mem=118.89 GB):  48%|████▊     | 28/58 [00:01<00:01, 22.62it/s]Capturing num tokens (num_tokens=512 avail_mem=118.87 GB):  48%|████▊     | 28/58 [00:01<00:01, 22.62it/s]Capturing num tokens (num_tokens=480 avail_mem=118.89 GB):  48%|████▊     | 28/58 [00:01<00:01, 22.62it/s]Capturing num tokens (num_tokens=448 avail_mem=118.89 GB):  48%|████▊     | 28/58 [00:01<00:01, 22.62it/s]Capturing num tokens (num_tokens=416 avail_mem=118.89 GB):  48%|████▊     | 28/58 [00:01<00:01, 22.62it/s]Capturing num tokens (num_tokens=384 avail_mem=118.88 GB):  48%|████▊     | 28/58 [00:01<00:01, 22.62it/s]Capturing num tokens (num_tokens=384 avail_mem=118.88 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.80it/s]Capturing num tokens (num_tokens=352 avail_mem=118.88 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.80it/s]Capturing num tokens (num_tokens=320 avail_mem=118.87 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.80it/s]Capturing num tokens (num_tokens=288 avail_mem=118.87 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.80it/s]

    Capturing num tokens (num_tokens=256 avail_mem=118.87 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.80it/s]Capturing num tokens (num_tokens=240 avail_mem=118.87 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.80it/s]Capturing num tokens (num_tokens=240 avail_mem=118.87 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.44it/s]Capturing num tokens (num_tokens=224 avail_mem=118.86 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.44it/s]Capturing num tokens (num_tokens=208 avail_mem=118.86 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.44it/s]Capturing num tokens (num_tokens=192 avail_mem=118.86 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.44it/s]Capturing num tokens (num_tokens=176 avail_mem=118.85 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.44it/s]Capturing num tokens (num_tokens=160 avail_mem=118.85 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.44it/s]Capturing num tokens (num_tokens=160 avail_mem=118.85 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.08it/s]Capturing num tokens (num_tokens=144 avail_mem=118.85 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.08it/s]Capturing num tokens (num_tokens=128 avail_mem=118.84 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.08it/s]Capturing num tokens (num_tokens=112 avail_mem=118.84 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.08it/s]

    Capturing num tokens (num_tokens=96 avail_mem=118.84 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.08it/s] Capturing num tokens (num_tokens=96 avail_mem=118.84 GB):  81%|████████  | 47/58 [00:01<00:00, 32.75it/s]Capturing num tokens (num_tokens=80 avail_mem=118.83 GB):  81%|████████  | 47/58 [00:01<00:00, 32.75it/s]Capturing num tokens (num_tokens=64 avail_mem=118.83 GB):  81%|████████  | 47/58 [00:02<00:00, 32.75it/s]Capturing num tokens (num_tokens=48 avail_mem=118.83 GB):  81%|████████  | 47/58 [00:02<00:00, 32.75it/s]Capturing num tokens (num_tokens=32 avail_mem=118.82 GB):  81%|████████  | 47/58 [00:02<00:00, 32.75it/s]Capturing num tokens (num_tokens=32 avail_mem=118.82 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.71it/s]Capturing num tokens (num_tokens=28 avail_mem=118.82 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.71it/s]Capturing num tokens (num_tokens=24 avail_mem=118.82 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.71it/s]

    Capturing num tokens (num_tokens=20 avail_mem=118.81 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.71it/s]Capturing num tokens (num_tokens=16 avail_mem=118.81 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.71it/s]Capturing num tokens (num_tokens=16 avail_mem=118.81 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.61it/s]Capturing num tokens (num_tokens=12 avail_mem=118.81 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.61it/s]Capturing num tokens (num_tokens=8 avail_mem=118.80 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.61it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=118.80 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.61it/s]Capturing num tokens (num_tokens=4 avail_mem=118.80 GB): 100%|██████████| 58/58 [00:02<00:00, 24.09it/s]


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
    Generated text:  Raul. I'm 17 years old and I live in Madrid. I'm a computer science student who is really passionate about technology. But what I'm really interested in is trying out new ways of learning. My teacher gave me a pair of Lego bricks and asked me to use them as a coding toy. I started coding in my spare time and have been doing so since the beginning of the year.  I have a good understanding of basic coding concepts. I have a very strong interest in the creation of new technology. I'm passionate about the way technology affects society. I have been doing this for almost a year now.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. She has a lot of responsibilities, but she also has fun. What would be the best title for this passage? The best title for this passage is __________. A) The President is a Very Important Person. B) The President has Lots of Responsibilities. C) The President Has Fun. D) The President Is Very Important.
    Answer: D) The President Is Very Important. The passage is discussing the importance of the president of the United States and their responsibilities, but also their enjoyment of the job. Therefore, the best title for this passage is "The President Is Very Important." The other options are not
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of Switzerland is Bern. Which is the capital of Switzerland?
    The capital of Switzerland is Bern. This is because Bern is the capital of Switzerland. The capital of France is Paris. The capital of Switzerland is Bern, not Paris. The order of capitals is: France > Germany > Switzerland > United Kingdom. So, among the options given, Bern is the capital of Switzerland. The correct answer is Bern. 
    
    However, to provide a comprehensive answer, it's worth noting that the capital of France is Paris, not Bern. Paris is the capital of France, while Bern is the capital of Switzerland. The correct answer to
    ===============================
    Prompt: The future of AI is
    Generated text:  here now.
    Artificial intelligence (AI) is changing the way we live, work, and communicate. Whether it’s robots navigating the city, self-driving cars that make the world safer, or AI-powered chatbots that can help you with any question, the future of AI is bright. As AI becomes more advanced and more prevalent, its impact on society will only grow.
    
    This is where the role of AI ethics comes into play. AI ethics is the study of how we should regulate and manage the use of AI, ensuring that it is used for the benefit of all. In this article, we will explore the role of AI ethics in


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. I enjoy [insert a short, positive description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new challenges and opportunities to grow and learn. What's your favorite book or movie? I'm always looking for new challenges and opportunities to grow and learn. What's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Academy of Sciences. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its cuisine, including its famous croissants and its traditional French wine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city that is both a living and a dying city, with its population aging and its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations, such as privacy, fairness, and accountability. This could lead to more robust regulations and standards for AI development and deployment.
    
    3. Greater reliance on machine learning: Machine learning is likely to become more prevalent in AI, allowing machines to learn and
    


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
    Generated text:  Emily, and I'm a writer. I've always loved writing, and I've always wanted to be a published author. I've been working on my books for years, and I've always had the passion and determination to create something truly great. I hope to turn my writing into a success, and I believe that my writing skills and love for storytelling are the keys to that success. Thanks for having me. That sounds great, Emily. You have a unique style that I really admire. Emily, can you give us a little bit about your writing process? Sure, I'm always looking for fresh ideas and fresh perspectives. My writing
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is the largest and most populous city in France, located in the French department of the Île-de-France. It is known for its rich history, art, fashion, and cuisine. Paris is a cultural and financial center with a rich history dating back to the Roman period. The city is also the capital of France and serves as the seat of government, parliament, and government offices. There are many attractions within Paris, such as the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and the Louvre Pyramid. It is also famous for its wine culture, including the famous Burgundy, Pinot
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but there are several trends that are likely to shape the technology's direction in the coming years:
    
    1. Increased collaboration between humans and machines: As AI technology continues to advance, we can expect to see a more interactive relationship between humans and machines. For example, chatbots and virtual assistants will become more intuitive and capable of providing personalized support and assistance.
    
    2. More ethical considerations: As AI systems become more advanced, there will likely be more attention paid to ethical considerations and the potential for unintended consequences. For example, AI-powered chatbots could be used to provide more accurate and useful information, but could also lead to biases and discrimination


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

    ].

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

     an

     [

    Your

     Profession

     or

     Career

    ]

     with

     [

    Your

     Job

     Title

    ]

     experience

    .

     I

     have

     been

     in

     the

     [

    Your

     Location

    ]

     for

     [

    Your

     Years

    ]

     years

    .

     My

     passion

     is

     [

    Your

     Passion

    ],

     and

     I

     am

     committed

     to

     [

    Your

     Commit

    ment

    ].

     I

     believe

     that

     with

     [

    Your

     Strength

    ],

     [

    Your

     Weak

    ness

    ],

     and

     [

    Your

     Challenges

    ],

     [

    Your

     Skills

    ]

     and

     [

    Your

     Strength

    s

    ],

     I

     can

     achieve

     [

    Your

     Goal

    ]

     and

     [

    Your

     Mission

    ].

     I

     am

     always

     looking

     for

     opportunities

     to

     [

    Your

     Goal

    ]

     and

     [

    Your

     Mission

    ].

     Thank

     you

     for

     having

     me

    .

     [

    Your

     Name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Petite

     France

    "

     or

     "

    La

     Grande

    ".

     It

     is

     the

     largest

     city

     in

     the

     country

     and

     the

     second

     largest

     by

     population

    ,

     after

     Paris

    ,

     in

     Europe

    .

     Paris

     is

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     many

     other

     historical

     and

     cultural

     landmarks

    .

     The

     city

     is

     also

     known

     for

     its

     fashion

    ,

     art

    ,

     and

     cuisine

    ,

     and

     is

     a

     popular

     tourist

     destination

    .

     Paris

     is

     home

     to

     approximately

     

    2

    2

     million

     residents

     and

     is

     often

     referred

     to

     as

     the

     "

    city

     of

     a

     thousand

     eyes

    ".

     The

     French

     capital

     has

     been

     a

     major

     center

     of

     culture

     and

     politics

     for

     over

     

    3

    0

    0

     years

     and

     remains

     one

     of

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     several

     trends

     are

     expected

     to

     drive

     advancements

     and

     changes

     in

     the

     field

    :
    


    1

    .

     Increased

     focus

     on

     ethics

     and

     fairness

    :

     As

     AI

     becomes

     more

     advanced

    ,

     we

     may

     see

     a

     growing

     focus

     on

     ethical

     and

     fairness

     concerns

    .

     This

     could

     lead

     to

     more

     stringent

     regulations

     and

     greater

     transparency

     in

     AI

     decision

    -making

     processes

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     AI

     is

     becoming

     increasingly

     integrated

     with

     other

     technologies

    ,

     such

     as

     blockchain

    ,

     machine

     learning

    ,

     and

     quantum

     computing

    ,

     creating

     a

     wider

     range

     of

     possibilities

     for

     AI

    -based

     applications

    .
    


    3

    .

     Universal

     AI

    :

     The

     possibility

     of

     creating

     an

     AI

     that

     can

     perform

     tasks

     that

     are

     currently

     being

     performed

     by

     humans

     has

     become

     increasingly

     possible

    .

     This

     could

     lead

    



```python
llm.shutdown()
```
