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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.48it/s]


    2026-05-09 03:48:34,928 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 03:48:34] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:12,  3.81it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:12,  3.81it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:12,  3.81it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:12,  3.81it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:12,  3.81it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:12,  3.81it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:12,  3.81it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:04<00:12,  3.81it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:04<00:12,  3.81it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:04<00:12,  3.81it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 22.39it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 22.39it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:04<00:00, 22.39it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:04<00:00, 31.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 43.11it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.42 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.57it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 35.57it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 35.57it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.76it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.76it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.76it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.76it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.76it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.76it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.12it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.12it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.90it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.90it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.90it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.90it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.90it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.90it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.44it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.44it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.44it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.44it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.44it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.44it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.92it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.92it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.92it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.92it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.92it/s] Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  81%|████████  | 47/58 [00:01<00:00, 31.12it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 31.12it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 31.12it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 31.12it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 31.12it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.87it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.13it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.13it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.13it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 34.46it/s]


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
    Generated text:  Christopher and I am a student at the University of California, Berkeley. I am now focusing on learning Chinese language and having a specialized in Computer Science. As a Chinese citizen, I have always felt fascinated by the vastness and complexity of Chinese culture, and its captivating way of expressing itself through literature, music, painting, and architecture. In addition to my academic pursuits, I am also passionate about volunteering and traveling. I am willing to immerse myself in Chinese culture and experience the local customs and traditions to gain a deeper understanding of the language and culture.
    
    I am also keen on developing my technical skills in areas such as machine learning, natural
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the following ________.
    A. Government
    B. Political party
    C. Church
    D. College
    Answer: A
    
    Given the function \( f(x) = 2x^2 + 3x - 1 \), use the bisection method to find the approximate root of the equation \( f(x) = 0 \) within the interval \([2, 3]\). What are the first two approximations?
    A. 1, 2
    B. 2, 2.5
    C. 1, 1.5
    D. 2, 2
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a city of many world famous attractions such as the Eiffel Tower, Notre Dame Cathedral, the Louvre Museum, the Arc de Triomphe, and so many others. Many people think that the capital of France is the most beautiful city in the world, but Paris has a lot to offer.
    France is the second most populous country in the world, and it is often referred to as "The City of Love" as well. Paris has a charming, lively and romantic atmosphere and a great range of places for people of all ages to visit. It has a long history dating back to ancient times, and has
    ===============================
    Prompt: The future of AI is
    Generated text:  here. This is a potentially game-changing technology that will impact almost every aspect of our lives. However, it is also a technology that is both revolutionary and complex. Understanding the role of AI in different fields of study and research is crucial to its potential to have a significant impact on society. This blog post will explore the different fields of study where AI is being used and what its potential to do for society.
    1. Artificial Intelligence (AI): AI is a branch of computer science that focuses on creating intelligent machines that can perform tasks that typically require human intelligence, such as learning, reasoning, and problem-solving. AI has been used in various


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am passionate about [reason for being at the company]. I am always looking for ways to [what I enjoy doing]. I am a [type of person] and I am always [what I enjoy doing]. I am [what I enjoy doing] and I am always [what I enjoy doing]. I am [what I enjoy doing] and I am always [what I enjoy doing]. I am [what I enjoy doing] and I am always [what I enjoy doing]. I am [what I enjoy doing] and I am always [what I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a cultural and economic hub of France and plays a significant role in the country's politics and economy. It is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. The city is also known for its food scene, with many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency, productivity, and cost savings for businesses.
    
    2. Enhanced personalization: AI will enable businesses to provide more personalized experiences to their customers. This will be achieved through the use of machine learning algorithms that can analyze customer data and provide tailored recommendations and offers.
    
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
    Generated text:  [Name] and I'm a [Career] [Job Title]. I have been a [Length of Service] [Years in Position]. I [Short Biography]. I am a dedicated [Role] [Position] who [Achievement/Continuation]. I have been [Length of Service] [Years in Position]. I am [Length of Service] [Years in Position]. I have a passion for [What I Do/What I Love]. I am a [Gender] [Race] [Nationality] [Age]. I am [Gender] [Race] [Nationality] [Age]. I am a [Gender] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A brief sentence elaborating on the geographical significance of Paris: Paris is known as the "City of Light" due to its vibrant culture, modern architecture, and appeal to tourists and locals alike. 
    
    Could you expand on the cultural attractions in Paris, such as art museums, historical sites, or famous landmarks? The city is home to several iconic landmarks and museums that attract visitors from around the world. Some of the most famous landmarks include the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. The city also hosts several museums, such as the Musée d'Orsay, the Musée Rod
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities and potential, but it is also filled with challenges and uncertainties. Here are some of the most promising trends in AI that could shape the future:
    
    1. Increased emphasis on ethical AI: As the need for ethical AI grows, companies and governments will be more likely to invest in AI that is designed to protect privacy, prevent bias, and promote fairness. This means that AI will become more transparent, accountable, and responsible.
    
    2. Deep learning and AI becoming more popular: As the cost of AI technology declines, the use of deep learning algorithms will become more common. These algorithms will be able to understand complex data patterns and


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

     fictional

     character

     name

    ],

     and

     I

     am

     [

    insert

     fictional

     character

     age

    ],

     [

    insert

     fictional

     character

     gender

    ],

     [

    insert

     fictional

     character

     profession

     or

     occupation

    ].

     I

     am

     a

     [

    insert

     fictional

     character

     profession

     or

     occupation

    ]

     and

     I

     have

     been

     in

     the

     [

    insert

     fictional

     character

     profession

     or

     occupation

    ]

     for

     [

    insert

     fictional

     character

     duration

     of

     time

    ],

     [

    insert

     fictional

     character

     profession

     or

     occupation

    ]

     to

     [

    insert

     fictional

     character

     duration

     of

     time

    ].

     Throughout

     my

     time

     in

     the

     field

    ,

     I

     have

     always

     been

     committed

     to

     [

    insert

     fictional

     character

     profession

     or

     occupation

    ],

     and

     I

     believe

     in

     [

    insert

     fictional

     character

     profession

     or

     occupation

    ]

     because

     [

    insert

     fictional

     character

     reason

    ].

     I

     believe

     that

     [

    insert

     fictional

     character

     reason

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ville

    -Mar

    ie

    "

     or

     simply

     "

    La

     Ville

    ,"

     and

     is

     the

     largest

     city

     in

     Europe

     by

     population

    ,

     with

     a

     population

     of

     over

     

    1

    0

     million

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     in

     the

     Mar

    ne

     department

     of

     the

     northern

     part

     of

     the

     country

    ,

     and

     is

     home

     to

     the

     presidential

     palace

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     cuisine

    ,

     and

     fashion

    ,

     and

     has

     been

     an

     important

     center

     of

     politics

     and

     culture

     in

     Europe

     for

     centuries

    .

     It

     is

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

     and

     is

     considered

     one

     of

     the

     most

     beautiful

     cities

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     far

     more

     transformative

     than

     ever

     before

    ,

     with

     many

     potential

     breakthrough

    s

     and

     opportunities

     on

     the

     horizon

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     integrated

     into

     society

    ,

     there

     will

     be

     an

     increased

     focus

     on

     ethical

     considerations

    ,

     such

     as

     ensuring

     that

     AI

     systems

     are

     fair

    ,

     transparent

    ,

     and

     have

     minimal

     bias

    .
    


    2

    .

     Greater

     reliance

     on

     data

    :

     As

     AI

     systems

     become

     more

     complex

     and

     sophisticated

    ,

     they

     will

     require

     more

     data

     to

     operate

     effectively

    .

     This

     will

     drive

     the

     need

     for

     greater

     investment

     in

     data

     collection

    ,

     analysis

    ,

     and

     storage

    .
    


    3

    .

     Continued

     advancements

     in

     machine

     learning

    :

     With

     the

     continued

     development

     of

     machine

    



```python
llm.shutdown()
```
