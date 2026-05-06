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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.49it/s]


    2026-05-06 10:53:05,367 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 10:53:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.40it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.40it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.40it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.40it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.40it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.40it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.40it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.40it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.40it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.40it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.40it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.15it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.15it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.15it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.15it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.15it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.15it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.15it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.15it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.26 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.37it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.10it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.10it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.93it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.93it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  21%|██        | 12/58 [00:00<00:01, 29.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.29it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.98it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.98it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.98it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.98it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.98it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.98it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.78it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.78it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.78it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.78it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.78it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.78it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=320 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=224 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.43it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.43it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.43it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.43it/s]

    Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.71it/s]Capturing num tokens (num_tokens=160 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.71it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.71it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.71it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.71it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.71it/s] Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=64 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 41.74it/s]

    Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  81%|████████  | 47/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.74it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.74it/s] Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=4 avail_mem=75.93 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=4 avail_mem=75.93 GB): 100%|██████████| 58/58 [00:01<00:00, 39.03it/s]


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
    Generated text:  Mark B, and I am a student at the University of Maryland, College Park, majoring in psychology. I have always been interested in a variety of topics, but I have always been more interested in the behavior and psychology of the young adult population. My field of interest is child and adolescent psychology, and I hope to develop my research skills by obtaining a Master's degree in child and adolescent psychology.
    I will be attending the University of Maryland's College Park campus in the fall of 2019.
    While attending college, I will be taking multiple courses in psychology, sociology, and English. I am also interested in getting involved
    ===============================
    Prompt: The president of the United States is
    Generated text:  scheduled to give a lecture on the first day of next month. He has invited several other important people to give lectures as well. If the president's lecture is the first lecture of the month, the other people will give the last lecture of the month. Each of these other people is scheduled to give a lecture for a week. How many lectures in total will there be in the month?
    
    To determine the total number of lectures in the month, we need to consider the following:
    
    1. The president's lecture is the first lecture of the month.
    2. Each of the other people will give a lecture for a week (7 days).
    
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the northwestern part of the country, close to the river Seine. The city is the largest metropolitan area in the country and the second largest in Europe. The city is the seat of a major political and economic power and is also a major educational center. The city is known for its impressive Notre-Dame Cathedral, famous for its distinctive stained glass windows and its status as one of the most important cultural landmarks of France.
    Can we draw the following conclusion? The capital of France is the largest city in the country.
    OPTIONS:
    A). yes;
    B). it is not possible to tell;
    C). no;
    A). Yes
    ===============================
    Prompt: The future of AI is
    Generated text:  an exciting journey ahead, with the potential to revolutionize virtually every industry and enhance our quality of life. But to achieve this, we need to develop and implement AI that is ethical and trustworthy. As an AI assistant, I can help you understand how AI can be used to make our lives better in the future.
    AI can be used to improve healthcare, such as through the use of AI algorithms that can detect diseases, predict patient outcomes, and assist doctors in diagnosis and treatment. Additionally, AI can be used to improve education by providing personalized learning experiences and teaching strategies that take into account an individual's strengths and weaknesses.
    AI can also be


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Vehicle] [Vehicle Name]. I have been driving for [Number of Years] years and have [Number of Miles] miles driven. I am a [Type of Vehicle] [Vehicle Name] and I have been driving for [Number of Years] years and have [Number of Miles] miles driven. I am a [Type of Vehicle] [Vehicle Name] and I have been driving for [Number of Years] years and have [Number of Miles] miles driven. I am a [Type of Vehicle] [Vehicle Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many of the world's most famous museums and attractions. Paris is a bustling metropolis with a rich history and a vibrant cultural scene, making it a popular tourist destination. The city is also known for its diverse cuisine and its role as a major financial center. Paris is a city of contrasts, with its modern architecture and high-tech industries blending seamlessly with its historic charm and cultural heritage. The city is also home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the potential for AI to replace human jobs.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, blockchain, and quantum computing. This will enable AI to perform tasks that are currently beyond its capabilities, such as autonomous
    


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
    Generated text:  [insert name here], and I'm a [insert profession here], by profession, as in, I work for a [insert company name here], and I've been with the company for [insert number here] years. I've always been passionate about [insert a personal interest or hobby here], and I'm always looking to learn and grow. I enjoy [insert a personal trait or quality here], and I'm always striving to do my best. I'm confident in my abilities, and I believe in the power of hard work and dedication to achieve my goals. Thank you for having me, [insert name here]. [insert name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its history, art, architecture, and food. Paris is often referred to as the "City of Light" and has a vibrant cultural scene. It is a major tourist destination and a major economy, with a large French-speaking population. The city is also home to many museums, theaters, and public parks. Paris is a popular destination for food lovers and enjoys being the culinary capital of Europe. The city is also an important center for French politics and has a history of political instability. Paris is a vibrant and diverse city with a rich culture, history, and modernity. It is known for its coffee,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly likely to include more advanced machine learning and natural language processing capabilities, while also exploring the use of AI in new and innovative ways. Some potential future trends include the development of more sophisticated AI systems that can interact with humans in ways that would be impossible today, the integration of AI into new forms of entertainment and messaging, the increasing use of AI in healthcare and other fields, and the development of more robust AI systems that can handle more complex and nuanced tasks.
    
    Additionally, as AI continues to advance, it is likely to become more integrated into our daily lives, with applications such as virtual assistants, smart home devices, and autonomous vehicles becoming increasingly


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

    'm

     a

     [

    insert

     occupation

     or

     profession

    ].

     I

    'm

     [

    insert

     how

     you

    'd

     like

     to

     be

     introduced

    ].

     I

    'm

     [

    insert

     your

     professional

     title

     or

     position

     if

     applicable

    ],

     and

     I

    've

     been

     working

     in

     this

     field

     for

     [

    insert

     number

     of

     years

    ]

     years

    .

     I

    'm

     always

     up

     to

     date

     with

     the

     latest

     trends

     and

     technologies

     in

     my

     industry

    ,

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     expand

     my

     hor

    izons

     and

     learn

     new

     skills

    .

     I

    'm

     a

     [

    insert

     one

     or

     two

     sentences

     that

     describe

     your

     personality

     or

     skills

    ].

     I

    'm

     [

    insert

     one

     or

     two

     sentences

     that

     describe

     your

     personality

     or

     skills

    ].

     I

    'm

     always

     [

    insert

     a

     verb

     that

     describes

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     and

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

     center

     of

     French

     culture

     and

     government

    .

     It

     is

     located

     in

     the

     southern

     part

     of

     the

     country

     on

     the

     banks

     of

     the

     Se

    ine

     river

     and

     is

     home

     to

     numerous

     famous

     landmarks

    ,

     including

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

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     known

     for

     its

     romantic

     and

     artistic

     atmosphere

    ,

     as

     well

     as

     its

     beautiful

     weather

     and

     vibrant

     festivals

    .

     With

     over

     

    3

    .

     

    7

     million

     inhabitants

    ,

     Paris

     is

     the

     second

     most

     populous

     city

     in

     the

     world

     after

     Mexico

     City

    .

     
    


    Paris

     is

     also

     known

     for

     its

     importance

     in

     fashion

    ,

     with

     many

     fashion

     designers

     and

     artists

     coming

     from

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     combination

     of

     exponential

     growth

     and

     rapid

     innovation

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     is

     already

     integrated

     into

     a

     wide

     range

     of

     devices

    ,

     from

     smartphones

     and

     laptops

     to

     smart

     homes

     and

     autonomous

     vehicles

    .

     As

     these

     technologies

     continue

     to

     evolve

    ,

     we

     can

     expect

     to

     see

     even

     more

     seamless

     integration

     of

     AI

     with

     other

     technologies

    .
    


    2

    .

     AI

     will

     become

     more

     diverse

     and

     personalized

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

     shift

     towards

     more

     diverse

     and

     personalized

     AI

     systems

    .

     These

     systems

     will

     be

     able

     to

     learn

     from

     multiple

     data

     sources

     and

     adapt

     to

     changing

     conditions

     in

     real

     time

    .
    


    3

    



```python
llm.shutdown()
```
