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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.77it/s]


    2026-05-12 02:04:05,910 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 02:04:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:56,  4.14s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:56,  4.14s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:56,  4.14s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:56,  4.14s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:56,  4.14s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.65it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.65it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.65it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.65it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.65it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.65it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.65it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.65it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.65it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.65it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.81it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.81it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.08it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.39it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.39it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.39it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.39it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.39it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.39it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.39it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.39it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.39it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.18 GB):   3%|▎         | 2/58 [00:00<00:03, 17.41it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.15 GB):   3%|▎         | 2/58 [00:00<00:03, 17.41it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.14 GB):   3%|▎         | 2/58 [00:00<00:03, 17.41it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.14 GB):   7%|▋         | 4/58 [00:00<00:03, 15.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.14 GB):   7%|▋         | 4/58 [00:00<00:03, 15.51it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.14 GB):   7%|▋         | 4/58 [00:00<00:03, 15.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.13 GB):   7%|▋         | 4/58 [00:00<00:03, 15.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.13 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.12 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.12 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.62it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=59.12 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.11 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.60it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.10 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.10 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.10 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.10 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.09 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.93it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=59.09 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.09 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.07 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.07 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.68it/s]Capturing num tokens (num_tokens=960 avail_mem=59.08 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.68it/s] Capturing num tokens (num_tokens=896 avail_mem=59.08 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.68it/s]Capturing num tokens (num_tokens=832 avail_mem=59.08 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.68it/s]Capturing num tokens (num_tokens=768 avail_mem=59.07 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.68it/s]Capturing num tokens (num_tokens=704 avail_mem=59.07 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.68it/s]Capturing num tokens (num_tokens=704 avail_mem=59.07 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=640 avail_mem=59.07 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=576 avail_mem=59.07 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.01it/s]

    Capturing num tokens (num_tokens=512 avail_mem=59.05 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=480 avail_mem=59.07 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=448 avail_mem=59.07 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=448 avail_mem=59.07 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=416 avail_mem=59.06 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=384 avail_mem=59.06 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=352 avail_mem=59.06 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=320 avail_mem=59.05 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.37it/s]Capturing num tokens (num_tokens=288 avail_mem=59.05 GB):  53%|█████▎    | 31/58 [00:01<00:00, 43.37it/s]

    Capturing num tokens (num_tokens=288 avail_mem=59.05 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.55it/s]Capturing num tokens (num_tokens=256 avail_mem=59.05 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.55it/s]

    Capturing num tokens (num_tokens=240 avail_mem=59.04 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.55it/s]Capturing num tokens (num_tokens=224 avail_mem=59.04 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.55it/s]Capturing num tokens (num_tokens=208 avail_mem=59.03 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.55it/s]Capturing num tokens (num_tokens=192 avail_mem=59.03 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.55it/s]Capturing num tokens (num_tokens=192 avail_mem=59.03 GB):  71%|███████   | 41/58 [00:01<00:00, 22.46it/s]Capturing num tokens (num_tokens=176 avail_mem=59.03 GB):  71%|███████   | 41/58 [00:01<00:00, 22.46it/s]Capturing num tokens (num_tokens=160 avail_mem=59.03 GB):  71%|███████   | 41/58 [00:01<00:00, 22.46it/s]Capturing num tokens (num_tokens=144 avail_mem=59.02 GB):  71%|███████   | 41/58 [00:01<00:00, 22.46it/s]Capturing num tokens (num_tokens=128 avail_mem=59.02 GB):  71%|███████   | 41/58 [00:01<00:00, 22.46it/s]Capturing num tokens (num_tokens=112 avail_mem=59.02 GB):  71%|███████   | 41/58 [00:01<00:00, 22.46it/s]Capturing num tokens (num_tokens=112 avail_mem=59.02 GB):  79%|███████▉  | 46/58 [00:01<00:00, 26.75it/s]Capturing num tokens (num_tokens=96 avail_mem=59.02 GB):  79%|███████▉  | 46/58 [00:01<00:00, 26.75it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=59.01 GB):  79%|███████▉  | 46/58 [00:01<00:00, 26.75it/s]Capturing num tokens (num_tokens=64 avail_mem=59.01 GB):  79%|███████▉  | 46/58 [00:01<00:00, 26.75it/s]Capturing num tokens (num_tokens=48 avail_mem=59.00 GB):  79%|███████▉  | 46/58 [00:01<00:00, 26.75it/s]Capturing num tokens (num_tokens=48 avail_mem=59.00 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=32 avail_mem=59.00 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=28 avail_mem=59.00 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=24 avail_mem=59.00 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=20 avail_mem=58.99 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=16 avail_mem=58.99 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=16 avail_mem=58.99 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.20it/s]Capturing num tokens (num_tokens=12 avail_mem=58.99 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.20it/s]

    Capturing num tokens (num_tokens=8 avail_mem=58.98 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.20it/s] Capturing num tokens (num_tokens=4 avail_mem=58.98 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.20it/s]Capturing num tokens (num_tokens=4 avail_mem=58.98 GB): 100%|██████████| 58/58 [00:01<00:00, 31.41it/s]


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
    Generated text:  Teda.
    
    I'm a young scientist with an interest in understanding the Earth and its inhabitants. I'm a leader of a team and I spend a lot of time playing with computers and experimenting with different algorithms to solve problems.
    
    Could you tell me about your work and how you became an AI model? I would be happy to hear about your background and how you came to be a leader in the field of AI.
    
    Sure, thank you for asking! As a young scientist, I've always been fascinated by the idea of understanding the Earth and its inhabitants. One of the biggest challenges in understanding our planet is how it works, how it is
    ===============================
    Prompt: The president of the United States is
    Generated text:  30 years older than the president of Brazil. The president of Brazil is 2 times older than the president of China. If the president of China will be 60 years old in 10 years, what is the current age of the president of China?
    To determine the current age of the president of China, we need to follow a step-by-step approach.
    
    1. **Find the current age of the president of China:**
       - The problem states that the president of China will be 60 years old in 10 years. Therefore, his current age is:
         \[
         60 -
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the largest city in the country.
    The answer is yes.
    The reasoning behind this answer is that Paris is the capital city of France, and it is known as the "City of Love" due to its famous romantic landmarks like Notre-Dame Cathedral and the Eiffel Tower. These features make Paris a popular destination for tourists and tourists' families, which would make it a popular place to live and visit. However, while Paris is a popular tourist destination, it is also a large city and not a small city like the capital of another country. Therefore, the statement "The capital of France is Paris" is accurate.
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be in smaller data sets, smaller organizations, and more interdisciplinary research. With these shifts in the industry, you will need a deep understanding of data and AI. To gain the knowledge and skills needed to support such an increase in the field, you will need to take a course or series of courses in the field.
    In this course, you will learn about the principles and processes of machine learning and deep learning, including how they relate to AI. You will also learn about the techniques used in AI, including algorithms, machine learning, and deep learning. You will also learn about the role of machine learning and deep learning in AI.
    


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about your personality or skills]. I enjoy [insert a short, positive, enthusiastic statement about your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What are your hobbies or interests? I enjoy [insert a short, positive, enthusiastic statement about your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What are
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and art scene. Paris is a cultural and economic hub of France and a major tourist destination. It is home to many world-renowned museums, theaters, and art galleries. The city is also known for its fashion industry, with many famous designers and boutiques. Paris is a city of contrasts, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the future.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology continues to improve, we can expect to see even more
    


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
    Generated text:  [insert name], and I'm a [insert your profession or role]. I recently graduated from [insert your college or university] with a degree in [insert your major]. I've always been passionate about [insert what you enjoy doing] and I've always been determined to achieve my goals. I'm always looking for ways to improve myself and I'm willing to take risks, even if it means starting a new career or going off on a journey. I'm confident in my abilities and I'm always looking for ways to help others succeed. I'm excited to be a part of a team and to make a positive impact on the world
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. This city is known for its rich history, beautiful architecture, and vibrant cultural scene. It is home to iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral, as well as a thriving arts community. Paris is also a major financial center and attracts millions of tourists each year. The city has a diverse population, with many different cultures and ethnicities living together.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several key trends, including:
    
    1. Increased use of machine learning and deep learning algorithms - This will drive the development of more accurate and efficient algorithms that can perform complex tasks such as natural language processing, image recognition, and predictive analytics.
    
    2. Integration of AI into everyday life - AI will become more integrated into our daily lives, from healthcare to transportation to customer service. This will allow us to interact with AI more easily and make more efficient use of resources.
    
    3. Greater focus on ethical considerations - As AI becomes more advanced, there will be an increased focus on addressing ethical concerns such as privacy, bias, and


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

     name

    ]

     and

     I

     am

     a

     [

    insert

     your

     occupation

    ]

     with

     a

     passion

     for

     [

    insert

     your

     area

     of

     interest

    ].

     I

     am

     always

     looking

     for

     ways

     to

     learn

     new

     things

     and

     explore

     new

     experiences

    ,

     and

     I

     am

     always

     eager

     to

     help

     others

     grow

     and

     learn

    .

     Whether

     you

    're

     a

     student

     or

     a

     professional

    ,

     I

     am

     here

     to

     listen and

     offer

     my

     guidance

     and

     support

    .

     What

     brings

     you

     to

     this

     moment

    ?

     I

     am

     [

    insert

     your

     age

    ],

     and

     I

     am

     here

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     How

     can

     I

     help

     you

     today

    ?

     I

     am

     [

    insert

     your

     profession

     or

     role

    ].

     Join

     me

     in

     my

     journey

     to

     become

     a

     better

     person

     and

     a

    
    
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

    ,

     located

     in

     the

     north

    western

     region

     of

     the

     country

    .

     It

     is

     one

     of

     the

     world

    's

     most

     important

     cultural

     and

     artistic

     centers

    ,

     known

     for

     its

     iconic

     landmarks

    ,

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

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     famous

     for

     its

     fashion

     industry

    ,

     its

     music

     scene

    ,

     and

     its

     annual

     celebrations

     of

     various

     holidays

     and

     events

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     and

     its

     rich

     cultural

     heritage

     and

     architectural

     diversity

     make

     it

     a

     major

     draw

     for

     visitors

     and

     locals

     alike

    .

     The

     city

     is

     known

     for

     its

     romantic

     atmosphere

    ,

     sophistication

    ,

     and

     charm

    ,

     which

     have

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     driven

     by

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     AI

     will

     continue

     to

     evolve

     and

     become

     more

     integrated

     into

     our

     daily

     lives

    .

     We

     may

     see

     more

     AI

    -powered

     tools

     and

     technologies

     in

     areas

     like

     healthcare

    ,

     education

    ,

     transportation

    ,

     and

     entertainment

    .
    


    2

    .

     There

     will

     be

     an

     increasing

     focus

     on

     ethical

     considerations

     and

     bias

     in

     AI

    .

     As

     AI

     becomes

     more

     sophisticated

    ,

     we

     may

     see

     more

     efforts

     to

     ensure

     that

     it

     is

     not

     being

     used

     to

     perpet

    uate

     or

     promote

     harmful

     biases

    .
    


    3

    .

     AI

     will

     be

     more

     integrated

     into

     existing

     technologies

     and

     systems

    .

     As

     AI

     continues

     to

     improve

    ,

     we

     may

     see

     more

     widespread

     adoption

     of

     AI

     in

     various

     industries

     and

     applications

    .
    


    4

    .

     There

     will

    



```python
llm.shutdown()
```
