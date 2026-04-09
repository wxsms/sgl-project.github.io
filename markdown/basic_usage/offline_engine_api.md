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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.37it/s]


    2026-04-09 01:06:38,853 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 01:06:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.52it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.52it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.52it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.52it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.52it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.52it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.52it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.52it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.52it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.71it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.71it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.71it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.71it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.71it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.71it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.71it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.71it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.68it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.68it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.68it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.68it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.68it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.68it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.68it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.68it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.24it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.14it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.14it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.14it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.14it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.14it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.14it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.14it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.50it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.50it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.50it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.50it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.50it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.50it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.50it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 44.59it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 44.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 17.42it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.42it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.42it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.56it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.56it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.56it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.56it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.05it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.05it/s]Capturing num tokens (num_tokens=960 avail_mem=120.29 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.05it/s] Capturing num tokens (num_tokens=960 avail_mem=120.29 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.91it/s]Capturing num tokens (num_tokens=896 avail_mem=120.28 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.91it/s]Capturing num tokens (num_tokens=832 avail_mem=119.96 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.91it/s]Capturing num tokens (num_tokens=768 avail_mem=119.05 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.91it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.91it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.91it/s]

    Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.71it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.71it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.71it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.71it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.71it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.71it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.67it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.67it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.67it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.67it/s]

    Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.67it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.01it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.31it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.31it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.31it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.31it/s]

    Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.31it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.31it/s] Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.81it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.81it/s]

    Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.81it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.81it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.81it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.76it/s] Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 36.20it/s]


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
    Generated text:  Alex Green and I am a private tutor with a passion for education. I have a degree in Mathematics and have been working as a tutor for more than 2 years. I love to help students understand math and get through difficult concepts. My goal is to make learning math as enjoyable as possible for my students. Let me know if you're interested in learning more about my services. 
    
    As a tutor, I value communication and patience because I want to help my students reach their full potential. I believe in setting clear goals and providing guidance to help students achieve their academic goals. Additionally, I believe in creating a positive learning environment for my students
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office with the task of forming a new administration and implementing policies to bring about change in the country. In a presidential election, there are three different candidates who run for the presidency, each with a different chance of winning the election. The probability of candidate A winning is 0.55, candidate B is 0.60, and candidate C is 0.35. If the election results are independent events, what is the probability that the new administration will not be the one with the highest number of seats in the new office of the president? To determine the probability that the new administration will not be the one
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a city and a historical city, located in the center of France. It is one of the most important cities in the world and Paris is one of the most visited cities in the world. The city is located on the banks of the Seine River.
    
    Paris is famous for its rich history, beautiful architecture, art, and cuisine. Paris is also famous for its fashion, music, dance, theater, and opera, especially in the city of Les 250, a famous fashion house. Paris is also famous for the opera house, the Opéra Garnier, and the Opera House.
    
    Paris is located in
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. AI will continue to grow in importance in many areas, including healthcare, finance, and entertainment. As technology continues to evolve, it is also becoming more complex and challenging for humans to understand and navigate. This makes it even more important to have a strong understanding of AI and its impact on society.
    In addition, the pace of development in AI is accelerating, and it will continue to improve in the coming years. This means that we will see more AI technologies that are not yet widely available, but are being developed and refined. These technologies will have a significant impact on various industries, including healthcare, finance, and entertainment.
    As AI


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


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to many world-renowned museums, including the Musée d'Orsay and the Musée Rodin. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also known for its fashion industry, with many famous designers and boutiques located in the city. The city is also home to the French Parliament building, which is the seat of the French government. Overall, Paris is a vibrant and diverse city with a rich history and a vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do. This could lead to more efficient and effective decision-making, as well as more personalized and context-aware interactions with humans.
    
    2. Greater emphasis on ethical and social implications: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social implications. This could lead to more careful design and development of AI systems, as well as greater consideration of the potential impact on society
    


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
    Generated text:  [Your Name], and I'm a [Your profession] who has always been fascinated by [Your hobby/interest]. I enjoy exploring new places and trying new things, always trying to push the boundaries of what's possible. I'm passionate about [Your passion/ambition], and I'm constantly learning and growing as a person. I love to share my knowledge and inspire others to do the same, and I'm always looking for new adventures and experiences. Thank you for having me. What is your hobby/interest? Hello, my name is [Your Name], and I'm a [Your profession] who has always been fascinated by
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Its most famous landmark is the Eiffel Tower, which stands as a symbol of the city. Paris is a world-renowned cultural and tourist destination, with many historical and artistic landmarks to explore. It is also known for its extensive array of cafes, bakeries, and other eateries. The city is renowned for its cuisine, particularly its famous French cuisine, which is heavily influenced by French culinary traditions. Additionally, Paris is home to the Louvre Museum, one of the world's most famous art museums. The city is also famous for its architecture, including its iconic Notre-Dame Cathedral and its beautiful gardens. Finally,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be heavily influenced by new advancements in machine learning, quantum computing, and the rise of blockchain technology. Here are some potential trends in AI that may become more prominent in the coming years:
    
    1. Increased integration with other fields: AI is becoming more integrated with other fields such as healthcare, finance, and education, allowing for more personalized and targeted solutions.
    
    2. Personalization: AI will allow for more personalized recommendations and experiences, as machines can learn and adapt to the user's behavior and preferences over time.
    
    3. Autonomous vehicles: The integration of AI in autonomous vehicles is likely to become more common in the coming years, allowing for


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

     I

    'm

     a

     [

    Occup

    ation

    ]

     with

     a

     passion

     for

     [

    Occup

    ation

    -related

     hobby

     or

     interest

    ].

     I

    'm

     [

    Age

    ]

     years

     old

    ,

     and

     I

     live

     in

     [

    City

     or

     Country

    ].

     I

    'm

     a

     [

    Occup

    ation

    ]

     who

     [

    Describe

     one

     or

     two

     aspects

     of

     your

     personality

     or

     character

    ].

     I

     love

     [

    Brief

    ly

     describe

     a

     memorable

     experience

     or

     situation

    ].

     And

     I

    'm

     always

     looking

     for

     [

    Brief

    ly

     describe

     a

     challenge

     or

     opportunity

     you

     are

     trying

     to

     overcome

    ].


    Hello

    ,

     my

     name

     is

     [

    Name

    ],

     I

    'm

     a

     [

    Occup

    ation

    ]

     with

     a

     passion

     for

     [

    Occup

    ation

    -related

     hobby

     or

     interest

    ].

     I

    'm

     [

    Age

    ]

     years

     old

    ,

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     rich

     history

    ,

     cultural

     attractions

    ,

     and

     iconic

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Latin

     Quarter

    .


    The

     capital

     of

     France

     is

     Paris

    ,

     a

     city

     known

     for

     its

     rich

     history

    ,

     cultural

     attractions

    ,

     and

     iconic

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Latin

     Quarter

    .

     The

     city

     is

     home

     to

     several

     museums

    , such

     as

     the

     Lou

    vre

     Museum

    ,

     where

     you

     can

     see

     works

     of

     art

     and

     artifacts

     from

     all

     over

     the

     world

    .

     The

     city

     also

     has

     a

     vibrant

     music

     scene

    ,

     with

     many

     famous

     artists

     and

     musicians

     coming

     to

     Paris

     each

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     difficult

     to

     predict

     with

     certainty

    ,

     but

     there

     are

     a

     number

     of

     possible

     trends

     that

     could

     potentially

     shape

     its

     direction

    .

     Some

     of

     the

     most

     likely

     trends

     include

    :
    


    1

    .

     The

     rise

     of

     large

    ,

     interconnected

     AI

     systems

    :

     As

     AI

     becomes

     more

     complex

     and

     distributed

    ,

     it

     is

     likely

     to

     form

     increasingly

     larger

     and

     interconnected

     systems

    .

     These

     systems

     could

     allow

     for

     greater

     efficiency

    ,

     scalability

    ,

     and

     collaboration

    ,

     but

     could

     also

     lead

     to

     greater

     risks

     of

     failure

     or

     mis

    interpret

    ation

    .
    


    2

    .

     AI

     becoming

     more

     human

    -like

    :

     As

     AI

     systems

     become

     more

     sophisticated

     and

     capable

     of

     learning

     and

     adapting

    ,

     it

     is

     possible

     that

     they

     will

     start

     to

     exhibit

     more

     human

    -like

     qualities

    .

     For

     example

    ,

     some

    



```python
llm.shutdown()
```
