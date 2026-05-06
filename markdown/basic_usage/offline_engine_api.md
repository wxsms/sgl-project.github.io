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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.10it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.09it/s]


    2026-05-06 22:05:02,497 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 22:05:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.13it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.13it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.13it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.29it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.29it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.29it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.29it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.29it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.29it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.29it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.29it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.61it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.61it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.61it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.61it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.61it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.61it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.61it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.61it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.61it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.61it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.64it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.64it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.64it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.64it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.64it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.64it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.64it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.64it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.64it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.64it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 20.64it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.82it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.82it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.82it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.82it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.82it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.82it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.82it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.82it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.82it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.82it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.82it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:05,  9.93it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:05,  9.93it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:05,  9.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:05,  9.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:03, 16.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:03, 16.10it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:03, 16.10it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:03, 16.10it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:03, 16.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.79it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  31%|███       | 18/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 33.19it/s] Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.80it/s]Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.80it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.80it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.80it/s]

    Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.80it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.80it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.11it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.11it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.11it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.11it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.11it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.11it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.45it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.45it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.45it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.45it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.45it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.45it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.80it/s]

    Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.18it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.18it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.18it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.18it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.18it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.18it/s] Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  81%|████████  | 47/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.86it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.86it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.86it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.86it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.86it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.86it/s] Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 34.59it/s]


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
    Generated text:  Bill. I'm in Grade 7 this year, and I have been doing math with my math teacher, Ms. Jones, for 10 years. She is very good at teaching math because she is very patient and understanding of the subject. She encourages her students to take their math abilities to the next level. I enjoy math class because I enjoy learning new things and solving new problems. I have a lot of fun with math. She also takes an interest in helping the students who are not sure what to do. She takes her students to the library and gives them a book that is about the mathematical principles of the book. The
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. There are some countries that he is willing to invade and another country that he is not willing to invade. At least half of the countries that he is willing to invade are also not willing to invade the country that he is not. There are 2 bases that the president has decided not to build because of security concerns. How many bases is the president considering building?
    Let's denote the number of countries the president is willing to invade as \( W \). According to the problem, at least half of the countries that he is willing to invade are also not willing to invade the country that he
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Rome
    D. Madrid
    Answer:
    
    A
    
    In the West, the President of the United States is elected through a ___ system.
    A. Indirect
    B. Direct
    C. Representative
    D. Consultative
    Answer:
    
    B
    
    Which of the following statements about the legal system of the United States is correct? A. The legal system of the United States has a complete legal system, covering all areas of society and all fields of human activity. B. The legal system of the United States only covers the fields of social and economic life. C. The legal system
    ===============================
    Prompt: The future of AI is
    Generated text:  more than clear
    For years, we've believed in the power of AI to bring much-needed progress to our society. More than a decade ago, we imagined a future where it would be an integral part of our daily lives. But now, we've entered a new phase in AI's journey.
    Although AI is still not mature enough to be used in all situations, we are on the verge of a breakthrough. But, we're not done. The future of AI is more than clear.
    Our society will be even more connected, and we will interact with technology on a daily basis. The world is in the midst of a digital revolution


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name],
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its rich history, beautiful architecture, and vibrant culture. It is also a major financial center and a major tourist destination. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its cuisine, including French cuisine and international cuisine. Paris is a city that is constantly evolving and changing, with new developments and attractions being added every year. The city is a cultural and economic hub in Europe and a major tourist destination for visitors from around the world. Paris is a city that is a true reflection of French culture and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs and preferences.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations and responsible use of AI. This could lead to more stringent regulations and guidelines for the development
    


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
    Generated text:  [Name], and I'm a [insert your occupation]. I've always been fascinated by the world around me, whether it's science, history, or technology. I enjoy exploring new topics, learning about different cultures, and appreciating the beauty of nature. I'm a problem solver, but I never give up on finding a solution that works. My drive is to help others and make the world a better place. If you have any questions or need help, don't hesitate to reach out. I look forward to the possibility of working with you. [Name] 
    
    Note: Replace [Name] with your actual name and [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    The statement is accurate. Paris, officially known as the Île-de-France, is the capital and largest city of France. It is the oldest continuously occupied city in the world and one of the world's most famous cities. Paris has a rich cultural history dating back over 2,500 years, with its historical sites, museums, and art scene attracting millions of visitors each year. It is known for its glamorous and diverse night life, world-renowned art collections, and numerous museums including the Louvre. Paris is also a major economic and financial hub, with its central business district and financial district, as well
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but some possibilities that could occur include:
    
      1. Increased integration of AI into various industries: AI could become more integrated into various industries, such as healthcare, finance, transportation, and manufacturing. This could lead to more efficient and effective use of resources, as well as better decision-making and operational effectiveness.
      2. AI becoming more autonomous: AI is becoming increasingly capable of performing tasks that previously required human intervention, such as driving a car or operating a business. This could lead to more autonomous vehicles, robots, and other AI-powered systems in the future.
      3. AI becoming more complex: As AI


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

     [

    Age

    ]

     year

    -old

     aspiring

     [

    Type

     of

     job

     or

     hobby

    ]

     [

    what

     it

     is

    ].

     I

     love

     [

    Favorite

     thing

     about

     [

    Job

    /H

    obby

    ]]

     because

     [

    why

     it

    's

     important

    ].

     I

     enjoy

     [

    what

     I

     enjoy

     doing

     that

     is

     outside

     of

     work

    ],

     [

    what

     makes

     it

     special

     to

     me

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     talk

     about

     [

    what

     you

     are

     excited

     to

     talk

     about

    ].

     I

     look

     forward

     to

     meeting

     you

     at

     [

    time

    /

    occasion

    ]

    !


    I

     hope

     you

     enjoy

     this

     short

    ,

     neutral

     self

    -int

    roduction

    !

     What

     do

     you

     think

    ?

     Is

     it

     too

     short

     or

     too

     long

    ?

     Are

     there

     any

     specific

     aspects

     that

     you

     feel

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    .

     It

     is

     a

     historic

     city

     with

     a

     rich

     history

     dating

     back

     to

     the

     Middle

     Ages

     and

     is

     the

     political

     and

     cultural

     center

     of

     the

     country

    .

     Paris

     is

     renowned

     for

     its

     art

    ,

     architecture

    ,

     and

     cuisine

     and

     is

     an

     important

     global

     center

     of

     culture

     and

     commerce

    .

     The

     city

     is

     home

     to

     numerous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     known

     for

     its

     vibrant

     nightlife

     and

     annual

     festivals

     such

     as

     the

     Christmas

     markets

    ,

     the

     Carnival

    ,

     and

     the

     Jazz

     Week

    .

     The

     city

     is

     an

     important

     hub

     of

     European

     culture

     and

     attracts

     millions

     of

     visitors

     each

     year

    .

     The

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     constantly

     evolving

    .

     Some

     possible

     trends

     in

     the

     coming

     years

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     assist

     with

     diagnosis

    ,

     treatment

     planning

    ,

     and

     drug

     discovery

    .

     We

     may

     see

     even

     more

     sophisticated

     applications

     in

     the

     future

    ,

     such

     as

     personalized

     medicine

     and

     predictive

     analytics

    .
    


    2

    .

     Improved

     AI

     ethics

    :

     With

     the

     increased

     use

     of

     AI

     in

     decision

    -making

     processes

    ,

     there

     will

     likely

     be

     greater

     emphasis

     on

     ethical

     considerations

     and

     accountability

    .

     This

     includes

     issues

     such

     as

     bias

    ,

     transparency

    ,

     and

     accountability

    .
    


    3

    .

     More

     AI

    -driven

     personal

     assistants

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     may

     see

     more

     AI

    -driven

     personal

     assistants

    



```python
llm.shutdown()
```
