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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.01it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.00it/s]


    2026-05-16 05:58:10,817 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 05:58:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.39it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.85it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.85it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.96it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.96it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.96it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.96it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.96it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.96it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.96it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.91 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.91 GB):   3%|▎         | 2/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.90 GB):   3%|▎         | 2/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.90 GB):   3%|▎         | 2/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.90 GB):   3%|▎         | 2/58 [00:00<00:02, 19.72it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.90 GB):   9%|▊         | 5/58 [00:00<00:02, 23.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.90 GB):   9%|▊         | 5/58 [00:00<00:02, 23.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.89 GB):   9%|▊         | 5/58 [00:00<00:02, 23.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.88 GB):   9%|▊         | 5/58 [00:00<00:02, 23.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.88 GB):   9%|▊         | 5/58 [00:00<00:02, 23.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.88 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.88 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.89it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.87 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.87 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.89it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.87 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.89it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=73.87 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.87 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.86 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.86 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.86 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.85 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.85 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.85 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.85 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.83 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.80it/s]Capturing num tokens (num_tokens=960 avail_mem=73.84 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.80it/s] Capturing num tokens (num_tokens=896 avail_mem=73.84 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.80it/s]

    Capturing num tokens (num_tokens=832 avail_mem=73.84 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.80it/s]Capturing num tokens (num_tokens=832 avail_mem=73.84 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=768 avail_mem=73.83 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=704 avail_mem=73.83 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=640 avail_mem=73.83 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=576 avail_mem=73.83 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=512 avail_mem=73.81 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=512 avail_mem=73.81 GB):  50%|█████     | 29/58 [00:00<00:00, 42.35it/s]Capturing num tokens (num_tokens=480 avail_mem=73.83 GB):  50%|█████     | 29/58 [00:00<00:00, 42.35it/s]Capturing num tokens (num_tokens=448 avail_mem=73.82 GB):  50%|█████     | 29/58 [00:00<00:00, 42.35it/s]

    Capturing num tokens (num_tokens=416 avail_mem=73.82 GB):  50%|█████     | 29/58 [00:00<00:00, 42.35it/s]Capturing num tokens (num_tokens=384 avail_mem=73.82 GB):  50%|█████     | 29/58 [00:00<00:00, 42.35it/s]Capturing num tokens (num_tokens=352 avail_mem=73.82 GB):  50%|█████     | 29/58 [00:00<00:00, 42.35it/s]Capturing num tokens (num_tokens=352 avail_mem=73.82 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.46it/s]Capturing num tokens (num_tokens=320 avail_mem=73.81 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.46it/s]Capturing num tokens (num_tokens=288 avail_mem=73.81 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.46it/s]

    Capturing num tokens (num_tokens=256 avail_mem=73.81 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.46it/s]Capturing num tokens (num_tokens=240 avail_mem=73.80 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.46it/s]Capturing num tokens (num_tokens=240 avail_mem=73.80 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=224 avail_mem=73.80 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=208 avail_mem=73.79 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=192 avail_mem=73.79 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=176 avail_mem=73.79 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=160 avail_mem=73.79 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=160 avail_mem=73.79 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=144 avail_mem=73.78 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=128 avail_mem=73.78 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.70it/s]

    Capturing num tokens (num_tokens=112 avail_mem=73.78 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=96 avail_mem=73.77 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.70it/s] Capturing num tokens (num_tokens=80 avail_mem=73.77 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=80 avail_mem=73.77 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.29it/s]Capturing num tokens (num_tokens=64 avail_mem=73.76 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.29it/s]Capturing num tokens (num_tokens=48 avail_mem=73.76 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.29it/s]Capturing num tokens (num_tokens=32 avail_mem=73.76 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.29it/s]Capturing num tokens (num_tokens=28 avail_mem=73.75 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.29it/s]Capturing num tokens (num_tokens=24 avail_mem=73.75 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.29it/s]Capturing num tokens (num_tokens=24 avail_mem=73.75 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=20 avail_mem=73.75 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.41it/s]

    Capturing num tokens (num_tokens=16 avail_mem=73.75 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=12 avail_mem=73.74 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=8 avail_mem=73.74 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.41it/s] Capturing num tokens (num_tokens=4 avail_mem=73.73 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=4 avail_mem=73.73 GB): 100%|██████████| 58/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=4 avail_mem=73.73 GB): 100%|██████████| 58/58 [00:01<00:00, 35.47it/s]


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
    Generated text:  Amin and I am a retired civil engineer and a computer programmer. I studied and worked at different universities, from the University of Michigan to the University of Hong Kong.
    How would you describe your professional background? I have worked for 27 years at different universities, both in civil and civil engineering. I have held various leadership positions, including the dean of the university and the chairman of the university. In my professional background, I have worked on many different projects, including civil engineering, environmental engineering, and urban planning. I have also worked on a wide range of projects in various fields, including healthcare and technology.
    In addition to my
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 36 years old. How old will the president be in 5 years if he has been married for 12 years, and he never gets pregnant or divorced by then?
    
    To determine the president's age in 5 years, we need to follow these steps:
    
    1. Identify the current age of the president.
    2. Add the number of years until he will be 16 years old.
    3. Confirm that he will not get pregnant or divorced by then.
    
    The president is currently 36 years old. He has been married for 12 years. Therefore, in 5 years, he will have married
    ===============================
    Prompt: The capital of France is
    Generated text:  [10].
    A. Paris
    B. London
    C. Moscow
    D. Beijing
    
    To determine the capital of France, let's review the capital cities mentioned in the options:
    
    1. Paris (France's capital)
    2. London (the capital of England)
    3. Moscow (the capital of Russia)
    4. Beijing (the capital of China)
    
    Given the options, Paris is the only capital city among the listed ones.
    
    Therefore, the capital of France is \boxed{A}.
    ===============================
    Prompt: The future of AI is
    Generated text:  predicted to be a blend of different technologies and methods, with a focus on human oversight and ethical considerations. How do you think AI will evolve in the coming years, and what are some of the key challenges that developers and researchers will face in this shift? In addition, what are the potential benefits and drawbacks of AI in various fields such as healthcare, finance, and education, and how can we ensure that the development and deployment of AI technologies are conducted in a responsible and ethical manner? Finally, what role can educators play in shaping the future of AI development and education? The future of AI is predicted to be a blend of different technologies and


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major economic and political center in Europe. It is also known for its fashion industry, art scene, and its role in the French language and culture. The city is home to many famous French artists, writers, and musicians, and is a popular tourist destination for visitors from around the world. Paris is a city of contrasts,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence. This could lead to more sophisticated forms of AI, such as those that can learn and adapt to new situations, and that can communicate with humans in a more natural way.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and
    


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
    Generated text:  [insert name] and I am a [insert age, gender, nationality, etc.] [insert occupation]. I am [insert degree, education, etc.]. I am passionate about [insert topic of interest, such as literature, music, art, or sports]. I love to [insert hobby, such as cooking, reading, or playing games]. I am always looking for [insert challenge, such as trying a new food, learning a new skill, or trying a new place to visit]. I love [insert activity, such as hiking, reading, or camping]. I am [insert any positive traits, such as [insert example
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    For a beginner, could you summarize the main idea of the statement in a single sentence? Paris is the capital of France. 
    
    The task involves a blend of creating a concise statement and summarizing the main idea in a single sentence. A concise statement captures the essential information, while a single sentence maintains the core meaning. The sentence must be straightforward and not overly complex, maintaining the core of the statement's meaning. Additionally, it should be easy to understand and convey the main idea accurately.
    You are an AI assistant that helps people find information. No product recommends this post. This post is not intended to be used for commercial
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several key trends, including:
    
    1. Increased focus on ethical considerations: There is a growing awareness of the ethical implications of AI, and there is a need for developers to ensure that AI systems are developed and used in ways that do not harm individuals or society as a whole.
    
    2. Greater reliance on machine learning and deep learning: As AI becomes more complex, it will require even more powerful algorithms and computational resources to perform tasks that were once feasible with traditional programming languages.
    
    3. Improved communication between humans and AI: There will be a need for humans to interact more closely with AI systems, as AI systems will require


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

    First

     Name

    ]

     [

    Last

     Name

    ],

     and

     I

    'm

     a

     [

    Occup

    ation

    ].

     I

     am

     passionate

     about

     [

    Why

     I

    'm

     passionate

    ].

     I

    'm

     [

    Age

    ],

     and

     I

     currently

     live

     in

     [

    Location

    ].

     I

     am

     [

    Gender

    ],

     and

     I

     have

     [

    Number

     of

     children

    ].

     My

     [

    Age

    ]

     is

     [

    Age

     Range

    ].

     I

     speak

     [

    Language

    ].

     I

     am

     [

    gender

    ]

     and

     I

     have

     a

     [

    Number

     of

     fingers

     and

     toes

    ]

     on

     my

     hands

    .

     I

     like

     [

    My

     favorite

     hobby

     or

     activity

    ].

     I

     love

     [

    My

     favorite

     book

     or

     movie

    ].

     My

     [

    Age

    ]

     is

     [

    Age

     Range

    ].

     My

     hobbies

     include

     [

    Number

     of

     hobbies

    ].

     I

     have

     a

     [

    Number

     of

     pets

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     and

     most

     populous

     city

     in

     France

     and

     the

     capital

     of

     France

    .

     Its

     population

     is

     over

     

    2

    .

    3

     million

     as

     of

     

    2

    0

    2

    1

    .

     It

     is

     located

     on

     the

     Se

    ine

     river

     and

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

    .

     It

     is

     an

     important

     cultural

    ,

     artistic

    ,

     and

     commercial

     center

     in

     France

     and

     is

     known

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

     Notre

    -D

    ame

     Cathedral

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

     a

     world

    -ren

    owned

     destination

     for

     tourists

    ,

     business

     travelers

    ,

     and

     entertainment

     venues

    .

     Despite

     its

     fame

    ,

     Paris

     remains

     a

     multicultural

     city

     with

     many

     different

     cultures

     and

     languages

     being

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     continued

     advancements

     in

     its

     complexity

    ,

     flexibility

    ,

     and

     applications

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

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

     becoming

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     big

     data

    ,

     machine

     learning

    ,

     and

     cloud

     computing

    .

     This

     integration

     is

     likely

     to

     continue

     as

     more

     companies

     and

     researchers

     focus

     on

     leveraging

     AI

     to

     solve

     complex

     problems

    .
    


    2

    .

     More

     personalized

     experiences

    :

     AI

     is

     already

     being

     used

     to

     provide

     more

     personalized

     experiences

     to

     customers

    ,

     such

     as

     through

     chat

    bots

     and

     voice

     assistants

    .

     The

     future

     of

     AI

     is

     likely

     to

     see

     even

     greater

     integration

     with

     customer

     data

     and

     behavior

     to

     provide

     even

     more

     personalized

     experiences

    .
    


    3

    .

     Enhanced

    



```python
llm.shutdown()
```
