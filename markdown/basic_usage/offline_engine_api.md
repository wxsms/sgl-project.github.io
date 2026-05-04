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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.18it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.18it/s]


    2026-05-04 05:06:59,665 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 05:06:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.06it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 14.92it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 14.92it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 14.92it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 14.92it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 14.92it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 14.92it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.92it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.96it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.96it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.92it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.67it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.67it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.67it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.67it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.77it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.77it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.77it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.77it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.77it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.77it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.26it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.43it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.43it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.43it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.43it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.43it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.43it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.43it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.48it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.48it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.64it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=16 avail_mem=72.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=16 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.08it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 41.90it/s]


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
    Generated text:  Trish and I am a professional graphic designer, illustrator, and book developer. My speciality is the creation of and editing of graphic designs and creative layouts for a variety of marketing materials. I have over 8 years of experience in graphic design and design work. I have worked with a wide range of clients from a diverse range of backgrounds and with a variety of projects of all sizes.
    I have completed work for a number of prominent brands including Denny's, The Munchies, and many others. I have also worked with a number of small businesses and freelance graphic designers.
    I have a passion for designing and developing digital content and
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to choose the person for the position of head of a federal agency. There are 100 candidates applying for the position, and each candidate has a unique score based on their qualifications and experience. The president wants to maximize the number of candidates selected for the position. However, the president realizes that if any two candidates have the same score, it is impossible to choose them both. 
    
    What is the maximum number of candidates the president can select to be the head of the federal agency?
    To maximize the number of candidates selected for the position, the president can select the candidates based on their scores. However, it is not possible to
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lille
    C. Lyon
    D. Lyon
    Answer:
    A
    
    For the same quality of material, which of the following sets of parameters will not change with the change of processing conditions?
    A. Yield strength
    B. Elastic modulus
    C. Tensile strength
    D. Compressive strength
    Answer:
    B
    
    Which of the following is NOT a primary function of construction project management?
    A. Coordinate project progress
    B. Organize resource allocation
    C. Reduce project cost
    D. Enhance project quality
    Answer:
    C
    
    A project manager is leading a team to create a
    ===============================
    Prompt: The future of AI is
    Generated text:  in development
    When the AI industry entered the public eye, it was a time when the technology was just beginning to show its potential. However, the industry is rapidly evolving and the new technology has the potential to become a transformative force.
    The technology has progressed quickly and the industry is steadily advancing in the world of AI. The public has seen a lot of advancements in recent years. The technology is growing and the industry is at the forefront of the development of the future.
    In the future, AI will become a critical part of our lives. As we learn more about how to use the technology, more and more people will be able to use


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. The city is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its annual festivals and events, including the Eiffel Tower Parade and the World Cup. The city is a major economic and cultural hub in Europe and plays a significant role in France's political and social life. Paris is a popular tourist destination and attracts millions of visitors
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be an increased focus on ethical considerations. This will include issues such as bias, transparency, accountability, and the impact of AI on society.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This will allow for more sophisticated and personalized interactions with AI-powered systems.
    
    3. Development of new AI technologies
    


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
    Generated text:  [Name], and I'm a [job title] for [company name]. I love [why you've chosen this job title], and I'm always looking for new ways to [what you do for a living]. I'm passionate about [what you do for a living] because [explain why you love this job and the work that you do]. I'm always looking for new opportunities to [what you want to be able to do on the job]. I'm eager to learn and grow and get to know [your company's culture and values], and I'm excited to contribute to [what you're passionate about doing on the job
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is a major city in France, known for its iconic landmarks, vibrant arts scene, and rich history. It is the second-largest city in France and the largest city by area. Paris is home to the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, among others. Its historical significance and modern influence make it one of the world's most popular cities. Paris is a cultural and commercial hub with a diverse population and a strong sense of French identity. Its vibrant nightlife and delicious cuisine attract millions of visitors each year. While Paris is known for its art, music, and fashion, it also has a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a number of factors, including advances in computing power, the development of new technologies, and the evolution of human society as a whole. Here are some possible future trends in AI:
    
    1. Increased emphasis on ethical considerations: As AI becomes more integrated into society, there will be increasing pressure to consider the ethical implications of its use. This will likely lead to a greater emphasis on ethical standards and principles, including the principles of fairness, transparency, and accountability.
    
    2. Greater reliance on machine learning: As AI becomes more advanced, it is likely to become more adept at learning from data and making decisions on its own.


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

     your

     full

     name

    ],

     and

     I

    'm

     [

    insert

     your

     occupation

     or

     profession

    ].

     I

     enjoy

     [

    insert

     one

     or

     two

     hobbies

     or

     passions

    ].

     I

     like

     to

     [

    insert

     something

     about

     myself

    ,

     such

     as

     how

     I

     stay

     healthy

    ,

     my

     favorite

     hobby

    ,

     etc

    .

    ].

     I

    'm

     always

     [

    insert

     something

     positive

     or

     helpful

     that

     you

     can

     say

     about

     me

    ,

     such

     as

     "

    ener

    getic

    ",

     "

    friendly

    ",

     etc

    .

    ].

     If

     you

     could

     name

     one

     thing

     you

     would

     do

     in

     the

     future

    ,

     it

     would

     be

     [

    insert

     one

     or

     two

     things

    ,

     such

     as

     "

    travel

    ",

     "

    read

    ",

     "

    write

    ",

     "

    travel

    ",

     "

    read

    ",

     "

    write

    ",

     etc

    .

    ].

     Thank

     you

     for

     asking

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     

    1

    2

    th

     largest

     city

     and

     the

     

    2

    nd

     most

     populous

     city

     in

     the

     European

     Union

    .

     It

     is

     the

     economic

     and

     cultural

     center

     of

     the

     country

     and

     home

     to

     the

     French

     government

    ,

     many

     of

     its

     museums

    ,

     and

     its

     iconic

     landmarks

    .

     It

     was

     founded

     by

     the

     Romans

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

     the

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

     other

     notable

     structures

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     with

     its

     many

     museums

    ,

     parks

    ,

     and

     other

     attractions

     attracting

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

     known

     for

     its

     rich

     culture

     and

     cuisine

    ,

     including

     its

     famous

     past

    ries

    ,

     wine

    ,

     and

     cuisine

    .

     Overall

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     involve

     a

     variety

     of

     trends

     and

     developments

    ,

     including

    :
    


    1

    .

     Greater

     integration

     of

     AI

     into

     all

     aspects

     of

     society

    ,

     including

     healthcare

    ,

     transportation

    ,

     energy

    ,

     and

     entertainment

    .
    


    2

    .

     Development

     of

     more

     advanced

     AI

     systems

     that

     can

     recognize

     and

     interpret

     complex

     natural

     language

    ,

     visual

    ,

     and

     audio

     data

    .
    


    3

    .

     Improved

     privacy

     and

     ethical

     considerations

     as

     AI

     systems

     become

     more

     integrated

     into

     daily

     life

    .
    


    4

    .

     Increased

     reliance

     on

     AI

     for

     decision

    -making

     in

     businesses

     and

     industry

    .
    


    5

    .

     Adv

    ancements

     in

     AI

     technology

     that

     can

     process

     vast

     amounts

     of

     data

     and

     analyze

     patterns

     that

     are

     difficult

     for

     humans

     to

     detect

    .
    


    6

    .

     Growing

     adoption

     of

     AI

     for

     personal

    ization

     in

     marketing

     and

     other

     forms

     of

    



```python
llm.shutdown()
```
