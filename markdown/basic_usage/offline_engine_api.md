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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.28it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]


    2026-05-15 16:47:28,485 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 16:47:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  3.99it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.95it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.95it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.09it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.09it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.09it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.09it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.09it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.09it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.09it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.09it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.09it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.09it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 22.42it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 22.42it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 22.42it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 22.42it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 22.42it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 22.42it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 22.42it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 22.42it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 22.42it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 22.42it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 22.42it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 31.79it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 31.79it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 31.79it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 31.79it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 31.79it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 31.79it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 31.79it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 31.79it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 31.79it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 31.79it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 31.79it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.85 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.82 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.82 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.81 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.81 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.81 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.81 GB):   9%|▊         | 5/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.81 GB):   9%|▊         | 5/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.80 GB):   9%|▊         | 5/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.79 GB):   9%|▊         | 5/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.79 GB):   9%|▊         | 5/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.79 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.79 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.78 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.78it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.68 GB):  21%|██        | 12/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.67 GB):  21%|██        | 12/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.67 GB):  21%|██        | 12/58 [00:00<00:02, 22.11it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.67 GB):  21%|██        | 12/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.67 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.93it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.67 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.66 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.66 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.66 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.65 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.65 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.64 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=960 avail_mem=58.65 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.81it/s] Capturing num tokens (num_tokens=896 avail_mem=58.65 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=832 avail_mem=58.64 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.81it/s]

    Capturing num tokens (num_tokens=768 avail_mem=58.64 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=768 avail_mem=58.64 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.18it/s]Capturing num tokens (num_tokens=704 avail_mem=58.64 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.18it/s]Capturing num tokens (num_tokens=640 avail_mem=58.63 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.18it/s]Capturing num tokens (num_tokens=576 avail_mem=58.63 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.18it/s]Capturing num tokens (num_tokens=512 avail_mem=58.62 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.18it/s]Capturing num tokens (num_tokens=480 avail_mem=58.63 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.18it/s]Capturing num tokens (num_tokens=480 avail_mem=58.63 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=448 avail_mem=58.63 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=416 avail_mem=58.63 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=384 avail_mem=58.63 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=352 avail_mem=58.62 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.85it/s]

    Capturing num tokens (num_tokens=320 avail_mem=58.62 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=320 avail_mem=58.62 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=288 avail_mem=58.61 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=256 avail_mem=58.61 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=240 avail_mem=58.61 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=224 avail_mem=58.60 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=208 avail_mem=58.60 GB):  60%|██████    | 35/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=208 avail_mem=58.60 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.09it/s]Capturing num tokens (num_tokens=192 avail_mem=58.60 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.09it/s]Capturing num tokens (num_tokens=176 avail_mem=58.60 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.09it/s]Capturing num tokens (num_tokens=160 avail_mem=58.59 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.09it/s]Capturing num tokens (num_tokens=144 avail_mem=58.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.09it/s]

    Capturing num tokens (num_tokens=128 avail_mem=58.13 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.09it/s]Capturing num tokens (num_tokens=128 avail_mem=58.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=112 avail_mem=57.66 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=96 avail_mem=55.07 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.37it/s] Capturing num tokens (num_tokens=80 avail_mem=55.07 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=64 avail_mem=55.07 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=48 avail_mem=55.06 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.37it/s]Capturing num tokens (num_tokens=48 avail_mem=55.06 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=32 avail_mem=55.06 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=28 avail_mem=55.06 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=24 avail_mem=55.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.84it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=16 avail_mem=55.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=16 avail_mem=55.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=12 avail_mem=55.04 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=8 avail_mem=55.04 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.29it/s] Capturing num tokens (num_tokens=4 avail_mem=55.04 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=4 avail_mem=55.04 GB): 100%|██████████| 58/58 [00:01<00:00, 35.05it/s]


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
    Generated text:  Eliza. I am a full-time mom who loves to explore the world of music with my children. I am from a family of musicians and I have had the opportunity to study music at a conservatory in a major city and to study with my father and my mother, who are both accomplished composers. I am also a professional musician in my own right, but I am not a music teacher and I hope to share my knowledge and love of music with others.
    
    When it comes to music, I have a deep passion and I love to learn about the history of music, the different genres, and different styles. I enjoy performing and performing
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to spend $50 million on a new missile defense system or $50 million on a new tax cut. He decides to run a poll to determine which option is more popular among the general public. He surveys a random sample of 400 voters and finds that 200 voters prefer the new missile defense system. Calculate the margin of error for the poll, assuming the confidence level is 95% and the sample size is 400.
    To calculate the margin of error for the poll, we need to use the formula for the margin of error in a proportion poll, which is given
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The city is famous for its fashion, its rich history and its beautiful architecture. It is the seat of the government of France and a center of culture, science and technology. Paris is also a major transportation hub, located on the Seine River, which flows through the city.
    Is there an answer to this question (If it cannot be answered, return "Unanswerable"). What is the capital of France? The capital of France is Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  coming! We are entering the age of automation and AI is transforming the way we live. This is why, AI is the future of the future. In today’s digital age, the focus of AI is on the creation of a digital platform that can predict the future, improve the quality of life for users, and generate a profit. The human being has always been a source of innovation, and the role of AI is to further develop this innovation. The role of AI is to assist humans in their day-to-day tasks and solve complex problems that require human creativity and expertise. AI is changing the way we live and work, and the world


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the Middle Ages. It is a major transportation hub and a major economic and financial center in Europe. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a popular tourist destination, with millions of visitors each year. It is also a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to become more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective AI systems that can handle a wider range of tasks.
    
    3. Greater reliance on AI for decision-making: AI is likely to
    


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
    Generated text:  [Your Name], and I'm a [Your Profession]. I'm a passionate and energetic person who thrives in a fast-paced environment. I love to explore new ideas and take on challenges. I enjoy mentoring and helping others achieve their goals. My work ethic is strong and I'm always looking for ways to improve my skills and knowledge. Overall, I'm very approachable and friendly, and I love making connections with people. I'm excited to get started with our conversation. Let's get to know each other better! [Your Name] (written in lowercase). How about you? What's your name, and what kind of work
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as "La Petite Lettouf".
    
    Paris is the capital city of France, and it is home to the Louvre Museum, Notre-Dame Cathedral, the Eiffel Tower, and other iconic landmarks. It is also a cosmopolitan and diverse city, known for its rich cultural heritage and romantic history. The city is home to a large French-speaking population, making it a popular destination for tourists and visitors from around the world. As the seat of France's government and capital, Paris plays a vital role in the country's economy, politics, and culture. With its rich history, stunning architecture, and vibrant culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends:
    
    1. Increased AI ethics: There is a growing awareness that AI has the potential to be harmful, and that human oversight is necessary to prevent its misuse. AI will likely become more transparent and accountable, and there will be more regulations and standards to ensure that AI is used ethically.
    
    2. Rise of AI superintends: As AI continues to develop, it is likely to become more capable and efficient, but it will also become more autonomous. AI superintends will be able to perform a wider range of tasks than current AI systems, such as planning and decision-making.
    
    3. AI


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

     [

    职业

    /

    成就

    ]

     with

     [

    Number

     of

     years

     in

     the

     field

    ].

     My

     skills

     include

     [

    skills

     and

     experiences

    ].

     I

     am

     currently

     a

     [

    current

     role

    ]

     in

     [

    company

     name

    ].

     And

    ,

     I

     am

     always

     looking

     to

     improve

     my

     [

    weak

    ness

     or

     skill

     area

    ].

     I

     am

     passionate

     about

     [

    why

     I

     am

     passionate

     about

     [

    field

    /

    area

     of

     focus

    ]],

     and

     I

     strive

     to

     [

    what

     I

     plan

     to

     do

     to

     improve

     my

     skills

    /

    perform

     better

     in

     this

     field

    /

    area

    ].

     Thank

     you

     for

     asking

    !

     

    🌍

    ✨

    
    


    ---
    


    Please

     modify

     the

     self

    -int

    roduction

     to

     include

     a

     description

     of

     your

     personal

     values

     and

     beliefs

     that

     align

     with

    
    
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

     Europe

     by

     population

     and

     has

     a

     rich

     cultural

     history

    .

     Paris

     is

     famous

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

     Lou

    vre

     Museum

    ,

     and

     Notre

     Dame

     Cathedral

    .

     It

     is

     also

     known

     for

     its

     café

     culture

     and

     world

    -ren

    owned

     cuisine

    .

     The

     city

    's

     architecture

    ,

     including

     the

     iconic

     E

    iff

    el

     Tower

    ,

     has

     been

     a

     symbol

     of

     France

     for

     over

     a

     century

    .

     Paris

     is

     often

     referred

     to

     as

     the

     "

    city

     of

     love

    "

     due

     to

     its

     romantic

     and

     romantic

     atmosphere

    .

     The

     city

     is

     also

     home

     to

     several

     important

     landmarks

     and

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

    .

     Overall

    ,

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     shaped

     by

     a

     number

     of

     complex

     trends

    ,

     including

    :
    


    1

    .

     Enhanced

     Natural

     Language

     Processing

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     understand

     and

     interpret

     human

     language

    ,

     leading

     to

     more

     sophisticated

     and

     intelligent

     chat

    bots

    ,

     virtual

     assistants

    ,

     and

     other

     natural

     language

     processing

     tools

    .
    


    2

    .

     Autonomous

     Vehicles

    :

     Self

    -driving

     cars

    ,

     trucks

    ,

     and

     airplanes

     will

     become

     more

     advanced

     and

     more

     common

    ,

     with

     AI

     systems

     that

     can

     navigate

     roads

     and

     highways

     with

     greater

     accuracy

     and

     efficiency

    .
    


    3

    .

     Voice

     Recognition

    :

     Voice

     recognition

     technology

     will

     continue

     to

     improve

    ,

     with

     AI

     systems

     that

     can

     accurately

     recognize

     and

     respond

     to

     voice

     commands

    ,

     such

     as

     controlling

     smart

     home

     devices

     or

     playing

     music

    .
    


    4

    .

     Improved

    



```python
llm.shutdown()
```
