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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.29it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.29it/s]


    2026-05-05 12:25:50,906 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 12:25:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.57it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.22it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.22it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 25.58it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 25.58it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 25.58it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 25.58it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 25.58it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 25.58it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 25.58it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 25.58it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 25.58it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:04<00:00, 25.58it/s]

    Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:04<00:00, 25.58it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 34.66it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 34.66it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 34.66it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 34.66it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 34.66it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 34.66it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 34.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.75 GB):   3%|▎         | 2/58 [00:00<00:02, 19.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:02, 19.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:02, 19.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:02, 19.75it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.74 GB):   9%|▊         | 5/58 [00:00<00:02, 23.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.74 GB):   9%|▊         | 5/58 [00:00<00:02, 23.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.73 GB):   9%|▊         | 5/58 [00:00<00:02, 23.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.72 GB):   9%|▊         | 5/58 [00:00<00:02, 23.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.72 GB):   9%|▊         | 5/58 [00:00<00:02, 23.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.71 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.71 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.71 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.17it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=73.71 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.69 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.69 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.67 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]Capturing num tokens (num_tokens=960 avail_mem=73.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s] Capturing num tokens (num_tokens=896 avail_mem=73.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]

    Capturing num tokens (num_tokens=832 avail_mem=73.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]Capturing num tokens (num_tokens=768 avail_mem=73.67 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.89it/s]Capturing num tokens (num_tokens=768 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=704 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=640 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=576 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=512 avail_mem=73.65 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=480 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=448 avail_mem=73.66 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=448 avail_mem=73.66 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.70it/s]Capturing num tokens (num_tokens=416 avail_mem=73.66 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.70it/s]Capturing num tokens (num_tokens=384 avail_mem=73.66 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.70it/s]Capturing num tokens (num_tokens=352 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.70it/s]

    Capturing num tokens (num_tokens=320 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.70it/s]Capturing num tokens (num_tokens=288 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.70it/s]Capturing num tokens (num_tokens=256 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.70it/s]Capturing num tokens (num_tokens=256 avail_mem=73.65 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=240 avail_mem=73.64 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=224 avail_mem=73.64 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=208 avail_mem=73.63 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=192 avail_mem=73.63 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=176 avail_mem=73.63 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=160 avail_mem=73.63 GB):  64%|██████▍   | 37/58 [00:01<00:00, 47.15it/s]Capturing num tokens (num_tokens=160 avail_mem=73.63 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.29it/s]Capturing num tokens (num_tokens=144 avail_mem=73.62 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.29it/s]Capturing num tokens (num_tokens=128 avail_mem=73.62 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.29it/s]

    Capturing num tokens (num_tokens=112 avail_mem=73.62 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.29it/s]Capturing num tokens (num_tokens=96 avail_mem=73.62 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.29it/s] Capturing num tokens (num_tokens=80 avail_mem=73.61 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.29it/s]Capturing num tokens (num_tokens=80 avail_mem=73.61 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.57it/s]Capturing num tokens (num_tokens=64 avail_mem=73.61 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.57it/s]Capturing num tokens (num_tokens=48 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.57it/s]Capturing num tokens (num_tokens=32 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.57it/s]Capturing num tokens (num_tokens=28 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.57it/s]Capturing num tokens (num_tokens=24 avail_mem=73.59 GB):  83%|████████▎ | 48/58 [00:01<00:00, 48.57it/s]Capturing num tokens (num_tokens=24 avail_mem=73.59 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.84it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.84it/s]Capturing num tokens (num_tokens=16 avail_mem=73.59 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.84it/s]

    Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.84it/s]Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.84it/s] Capturing num tokens (num_tokens=4 avail_mem=73.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.84it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:01<00:00, 43.52it/s]


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
    Generated text:  Laura and I am a writer, a teacher, and a mom. I have just published two books, "The Mysterious Dreams of the Unshaven" and "The Greatness of Mystery". Both books deal with themes of spirituality, the subconscious, and how to break free of the shame of the mind. I am currently teaching at a school in Spain. What inspired you to write these books? What do you hope students will learn from your books? How do your books connect with students' lives?
    Laura: Thanks for asking! I've always loved writing, and I have a degree in English and English Literature from California State University
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 3 inches tall. If someone is 3 feet shorter than the president, how tall is the person? To determine the height of the person who is 3 feet shorter than the president, we need to follow these steps:
    
    1. Convert the president's height from feet and inches to just feet.
    2. Subtract 3 feet from the president's height to find the height of the person who is 3 feet shorter.
    
    First, let's convert the president's height from feet and inches to just feet. The president is 5 feet 3 inches tall. Since there are 12 inches in a foot,
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A) Paris
    B) Lyon
    C) Lille
    D) Bordeaux
    
    To determine the capital of France, let's analyze each option step by step.
    
    1. **Paris**: Paris is the capital of France. It is the largest city in France and the seat of the government, as well as the main cultural and economic center of the country. So, Paris is the correct capital of France.
    
    2. **Lyon**: Lyon is a city in the Loire Valley region of France, but it is not the capital of France. Lyon is the capital of the department of Val-de-Loire and is located in
    ===============================
    Prompt: The future of AI is
    Generated text:  fast approaching and as a business professional, it’s imperative that you’re able to implement the latest AI technologies in a way that ensures your company is competitive and ahead of the curve. But where do you start? What skills do you need to build a robust and efficient AI solution that will revolutionize the way you do business? Let’s explore the top skills you’ll need to build a robust AI solution that will revolutionize your company.
    
    1. Strong Credibility and Trust
    
    First and foremost, you need to build a reputation for being a reliable and trustworthy company. Make sure that your team is well-trained and well-equipped to handle the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few details about your personality, skills, or interests that make you stand out]. And what's your background? I've always been interested in [insert a few details about your hobbies, interests, or experiences that make you unique]. And what's your favorite hobby or activity? I love [insert a few details about your favorite hobby or activity]. And what's your favorite book or movie? I love [insert a few details about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French language. Paris is also home to many famous French artists, writers, and musicians. The city is known for its cuisine, including its famous pastries and its traditional French dishes. Paris is a vibrant and dynamic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This integration will allow AI to perform tasks that are currently performed by humans, such as image recognition, speech recognition, and autonomous driving.
    
    2. Enhanced privacy and security: As AI becomes more integrated with other technologies, there will be increased concerns about privacy and security. This will lead to more stringent regulations and standards for AI development and use.
    
    
    


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
    Generated text:  Alex, and I am a skilled and experienced software engineer. I have a passion for innovation and am always looking for ways to improve existing systems and technologies. I have a strong work ethic and enjoy working in a fast-paced environment. I am a team player and am always willing to collaborate with others to achieve our goals. I am also a great communicator and am always eager to help others. Overall, I am a proactive and enthusiastic person who is always looking for ways to contribute to the success of our team. Thank you. 
    
    ---
    
    If I could summarize what Alex does and say something about the person he is, what would you say?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its rich history, beautiful architecture, and iconic landmarks such as Notre-Dame Cathedral and the Eiffel Tower. Paris is also famous for its fashion, art, and cuisine. The city is home to over 3 million people and is a major tourist destination. Its 13th and 14th centuries Gothic Quarter and 17th century Notre-Dame Cathedral are must-see attractions. The city is also known for its political and social challenges, including its role as a major European power during the French Revolution and World Wars. Paris is a vibrant, dynamic city that continues to be a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be a mix of progress, innovation, and potential setbacks. Here are some of the possible trends that we can expect:
    
    1. Increased proficiency of AI: AI technology will continue to improve and become more accurate, leading to the development of more complex and sophisticated AI systems. This will make AI systems more capable of solving complex problems and improving their performance in a wide range of applications.
    
    2. Enhanced personalization: As AI technology continues to improve, we can expect to see a more personalized experience for users. This will be achieved through the use of big data and machine learning to provide personalized recommendations and services.
    
    3. Autonomous vehicles:


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

    Character

    's

     Name

    ].

     I

    'm

     a

     [

    Title

    ]

     with

     [

    Number

    ]

     of

     years

     of

     experience

     in

     [

    Field

     of

     Interest

    ].

     I

     started

     my

     career

     in

     [

    Start

     Company

     Name

    ]

     and

     quickly

     gained

     success

     in

     [

    Field

     of

     Interest

    ].

     I

    'm

     now

     [

    Current

     Position

    ]

     at

     [

    Company

     Name

    ].

     In

     my

     spare

     time

    ,

     I

     enjoy

     [

    Inter

    ests

    /

    Activities

    ].

     Let

     me

     know

     if

     you

    're

     interested

     in

     learning

     more

     about

     me

    !

     [

    Character

    's

     Name

    ]

     


    (

    You

     can

     add

     your

     own

     details

    ,

     like

     your

     name

    ,

     company

    ,

     and

     hobbies

    )


    Hello

    ,

     my

     name

     is

     [

    Character

    's

     Name

    ].

     I

    'm

     a

     [

    Title

    ]

     with

     [

    Number

    ]

     of

     years

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     


    (

    If

     the

     question

     is

     un

    answer

    able

    ,

     state

     "

    Un

    answer

    able

    ").

     Un

    answer

    able

    .

     Paris

     is

     the

     capital

     of

     France

    .

     The

     statement

     is

     factual

     and

     accurate

    .

     However

    ,

     it

     could

     be

     made

     more

     precise

     by

     specifying

     that

     Paris

     is

     the

     capital

     of

     France

     and

     that

     it

     is

     also

     the

     largest

     city

     in

     the

     country

    .

     It

     would

     also

     be

     helpful

     to

     provide

     additional

     context

     about

     the

     significance

     of

     Paris

     as

     a

     cultural

     and

     economic

     center

     of

     France

    .

     
    


    Additional

     information

    :


    -

     Paris

     is

     the

     second

     largest

     city

     in

     France

     and

     the

     second

    -largest

     city

     in

     Europe

    .


    -

     It

     is

     the

     seat

     of

     government

    ,

     the

     country

    's

     legislative

    ,

     executive

    ,

     and

     judicial

     branches

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     and

     there

     are

     several

     areas

     where

     we

     can

     expect

     significant

     advancements

     and

     potential

     innovations

    :
    


    1

    .

     Self

    -driving

     cars

    :

     As

     autonomous

     technology

     improves

     and

     becomes

     more

     accessible

    ,

     we

     can

     expect

     to

     see

     widespread

     adoption

     of

     self

    -driving

     cars

     in

     both

     public

     and

     private

     vehicles

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

     artificial

     intelligence

     to

     navigate

     streets

     and

     highways

     with

     greater

     ease

     and

     efficiency

    .
    


    2

    .

     Medical

     imaging

    :

     AI

     will

     be

     used

     to

     improve

     the

     accuracy

     and

     efficiency

     of

     medical

     imaging

    ,

     such

     as

     X

    -rays

     and

     MR

    Is

    .

     AI

     algorithms

     will

     be

     able

     to

     analyze

     medical

     images

     faster

     and

     with

     greater

     precision

    ,

     leading

     to

     more

     accurate

     diagnoses

     and

     treatment

     outcomes

    .
    


    3

    .

    



```python
llm.shutdown()
```
