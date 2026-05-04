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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.77it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.76it/s]


    2026-05-04 06:59:28,724 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 06:59:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:41,  4.93s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:41,  4.93s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.58it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.06it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 13.00it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 13.00it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 13.00it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 13.00it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 13.00it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.00it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.00it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.00it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.00it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 19.00it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 25.83it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 25.83it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 25.83it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 25.83it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 25.83it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 25.83it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 25.83it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 25.83it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 25.83it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:05<00:00, 25.83it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 34.56it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 34.56it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 34.56it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 34.56it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 34.56it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 34.56it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.16it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.66 GB):   3%|▎         | 2/58 [00:00<00:03, 17.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.66 GB):   3%|▎         | 2/58 [00:00<00:03, 17.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.66 GB):   3%|▎         | 2/58 [00:00<00:03, 17.57it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.66 GB):   3%|▎         | 2/58 [00:00<00:03, 17.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.66 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.65 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.64 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.64 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.64 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=53.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.63 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.62 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.62 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.62 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.62 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.61 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.61 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.61 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.61 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.71it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=53.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.71it/s]Capturing num tokens (num_tokens=960 avail_mem=53.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.71it/s] Capturing num tokens (num_tokens=960 avail_mem=53.60 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.30it/s]Capturing num tokens (num_tokens=896 avail_mem=53.60 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.30it/s]Capturing num tokens (num_tokens=832 avail_mem=53.59 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.30it/s]Capturing num tokens (num_tokens=768 avail_mem=53.59 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.30it/s]Capturing num tokens (num_tokens=704 avail_mem=53.59 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.30it/s]Capturing num tokens (num_tokens=640 avail_mem=53.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.30it/s]Capturing num tokens (num_tokens=640 avail_mem=53.58 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.69it/s]Capturing num tokens (num_tokens=576 avail_mem=53.58 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.69it/s]

    Capturing num tokens (num_tokens=512 avail_mem=53.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.69it/s]Capturing num tokens (num_tokens=480 avail_mem=53.58 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.69it/s]Capturing num tokens (num_tokens=448 avail_mem=53.58 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.69it/s]Capturing num tokens (num_tokens=416 avail_mem=53.58 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.69it/s]Capturing num tokens (num_tokens=416 avail_mem=53.58 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=384 avail_mem=53.58 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=352 avail_mem=53.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=320 avail_mem=53.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=288 avail_mem=53.56 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=256 avail_mem=53.56 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.08it/s]

    Capturing num tokens (num_tokens=256 avail_mem=53.56 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.90it/s]Capturing num tokens (num_tokens=240 avail_mem=53.56 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.90it/s]Capturing num tokens (num_tokens=224 avail_mem=53.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.90it/s]Capturing num tokens (num_tokens=208 avail_mem=53.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.90it/s]Capturing num tokens (num_tokens=192 avail_mem=53.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.90it/s]Capturing num tokens (num_tokens=176 avail_mem=53.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.90it/s]Capturing num tokens (num_tokens=176 avail_mem=53.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=160 avail_mem=53.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.51it/s]

    Capturing num tokens (num_tokens=144 avail_mem=53.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=128 avail_mem=53.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=112 avail_mem=53.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=96 avail_mem=53.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.51it/s] Capturing num tokens (num_tokens=96 avail_mem=53.53 GB):  81%|████████  | 47/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=80 avail_mem=53.53 GB):  81%|████████  | 47/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=64 avail_mem=53.52 GB):  81%|████████  | 47/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=48 avail_mem=53.52 GB):  81%|████████  | 47/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=32 avail_mem=53.52 GB):  81%|████████  | 47/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=28 avail_mem=53.51 GB):  81%|████████  | 47/58 [00:01<00:00, 36.27it/s]

    Capturing num tokens (num_tokens=28 avail_mem=53.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 37.98it/s]Capturing num tokens (num_tokens=24 avail_mem=53.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 37.98it/s]Capturing num tokens (num_tokens=20 avail_mem=53.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 37.98it/s]Capturing num tokens (num_tokens=16 avail_mem=53.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 37.98it/s]Capturing num tokens (num_tokens=12 avail_mem=53.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 37.98it/s]Capturing num tokens (num_tokens=12 avail_mem=53.50 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=8 avail_mem=53.50 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.83it/s] Capturing num tokens (num_tokens=4 avail_mem=53.50 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.83it/s]Capturing num tokens (num_tokens=4 avail_mem=53.50 GB): 100%|██████████| 58/58 [00:01<00:00, 35.67it/s]


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
    Generated text:  Daniel J. Rubin and I am a professor of medicine in the Department of Psychiatry and Behavioral Science at the University of Pennsylvania School of Medicine. I have been conducting research in the neurology clinic at the Philadelphia Veterans Home since 1983, working with a team of neurologists and neurosurgeons to study the impacts of traumatic brain injury and to develop novel therapeutic approaches for severe neurological conditions. In addition to being the director of the Penn Center for Trauma Research, I have served as chief of the Division of Neurology at the University of Pennsylvania, director of the Penn Center for Trauma Research, and associate dean for
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 4 inches tall. If the average height of a U. S. adult is 5 feet 7 inches, how many adult feet tall is the president? Express your answer as a decimal to the nearest tenth.
    
    To determine the height of the president in feet, we first need to convert the president's height from inches to feet. The president's height is given as 5 feet 4 inches. Since there are 12 inches in a foot, we can convert the inches to feet by dividing 4 inches by 12:
    
    \[ 5 \text{ feet} + 4 \text{ inches
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the western coast of the Mediterranean. It is located at the northern tip of the island of Corsica. The capital is the city of Paris.
    This is the second most populous city in France, after Paris. The city has a population of 2.15 million (2019) based on the 2019 census.
    The capital of France is Paris. The capital is also known as "La République" (French: "La République", lit. "The Republic") and it is often also known as "Paris". The most commonly accepted French pronunciation of "Paris" is /
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of researchers. If you're interested in AI and want to learn about the latest research and developments, this book is a great place to start. The book is written in a clear and concise style, with plenty of engaging examples and real-world applications to illustrate the concepts covered. It's also updated frequently with the latest research findings and insights. If you're looking for a book that will keep you entertained and informed for years to come, then this is definitely a must-read. The book is well-organized and easy to navigate, with each chapter providing a clear introduction to the topic of AI and a detailed discussion of the latest


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Person] who is [Describe your personality traits here]. I have a [Describe your hobbies or interests here]. I am [Describe your strengths and weaknesses here]. I am [Describe your goals and aspirations here]. I am [Describe your future plans here]. I am [Describe your life style here]. I am [Describe your personality type here]. I am [Describe your physical appearance here]. I am [Describe your physical abilities here]. I am [Describe your physical appearance here]. I am [Describe your physical abilities here]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is located on the Seine River and is the seat of government, administration, and culture for the French Republic. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, music, and literature. Paris is a cultural and economic hub of the world and is a major tourist destination. It is home to many famous museums, theaters, and other cultural institutions. The city is also known for its cuisine,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more efficient and effective AI systems that can better understand and respond to human needs and preferences.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be a need to address privacy and security concerns. This could lead to the development of new technologies and approaches to protect user data and prevent cyber attacks.
    
    3. Increased reliance on AI for decision-making: AI is likely to become more integrated
    


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
    Generated text:  [Name]. I'm an experienced computer programmer with over 5 years of experience in the field, and I have a passion for staying up-to-date with the latest software development trends and technologies. I'm also a master of debugging and code review, and have helped countless developers improve their coding skills and write better code. I'm a natural problem-solver and am always looking for ways to learn and grow in my field. I'm excited to be here and I'm happy to assist you with your coding needs. What is your name? [Name]
    I'm a skilled programmer with over 5 years of experience, and I have a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    [1] The city is situated on the banks of the river Seine, which flows through the center of the city. 
    
    [2] It is home to many world-renowned landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.
    
    [3] The city has a rich history dating back to the 6th century, with its historical center located in the Seine Valley.
    
    [4] Today, Paris is a cosmopolitan metropolis with a diverse population and a strong economy, known for its culture, food, and fashion. 
    
    [5] It is also a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  complex and unpredictable, but there are some general trends that are likely to shape the industry as we know it. Some potential future trends in AI include:
    
    1. Increased integration with everyday life: AI is increasingly being integrated into our everyday lives, from smart home technology to self-driving cars. This trend is expected to continue as more companies invest in AI and integrate it into their products and services.
    
    2. Greater focus on ethical and legal issues: The development of AI raises important ethical and legal questions that need to be addressed. Governments, businesses, and consumers are all taking steps to ensure that AI is used responsibly and in a way that respects privacy


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

    current

     profession

     or

     area

     of

     expertise

    ]

     with

     a

     passion

     for

     [

    mot

    iv

    ational

     quote

    ].

     I

    'm

     excited

     to

     share

     my

     experiences

     and

     learn

     new

     things

     every

     day

    .

     I

     enjoy

     participating

     in

     group

     activities

     and

     reading

     motivational

     books

    .

     I

     also

     love

     traveling

     and

     exploring

     new

     places

    .

     My

     dream

     is

     to

     achieve

     my

     long

    -term

     goals

     and

     make

     a

     positive

     impact

     on

     the

     world

    .

     I

    'm

     always

     ready

     to

     learn

     and

     grow

    ,

     and

     I

     look

     forward

     to

     meeting

     new

     people

     and

     making

     new

     connections

    .

     What

    's

     your

     name

     and

     what

     do

     you

     do

    ?

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     a

     personal

     name

    ,

     but

     my

     primary

     function

     is

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

    ,

     located

     in

     the

     south

     of

     the

     country

     and

     on

     the

     Atlantic

     coast

    .

     It

     is

     known

     for

     its

     beautiful

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     culture

    .

     The

     city

     is

     home

     to

     many

     famous

     landmarks

     such

     as

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

     a

     bustling

     met

    ropolis

     with

     a

     population

     of

     over

     

    1

     million

     people

     and

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

    .

     Its

     economy

     is

     heavily

     reliant

     on

     tourism

    ,

     making

     it

     an

     important

     economic

     center

     in

     France

     and

     across

     Europe

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     including

     the

     Chanel

    ,

     Louis

     V

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     difficult

     to

     predict

    .

     However

    ,

     some

     potential

     trends

     that

     could

     influence

     its

     development

     include

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

     algorithms

     and

     techniques

    :

     As

     machine

     learning

     algorithms

     become

     more

     sophisticated

    ,

     AI

     systems

     could

     become

     even

     more

     capable

     in

     recognizing

     patterns

     and

     making

     predictions

    .

     This

     could

     lead

     to

     breakthrough

    s

     in

     areas

     such

     as

     autonomous

     vehicles

    ,

     self

    -driving

     cars

    ,

     and

     even

     virtual

     assistants

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     AI

     will

     likely

     become

     more

     integrated

     with

     other

     technologies

     such

     as

     sensors

    ,

     computers

    ,

     and

     the

     internet

    .

     This

     could

     lead

     to

     smarter

     devices

     and

     systems

     that

     are

     more

     efficient

     and

     reliable

    .
    


    3

    .

     More

     ethical

     concerns

    :

     As

     AI

     becomes

     more

     advanced

    ,

     there

     will

    



```python
llm.shutdown()
```
