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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.29it/s]


    2026-05-09 06:31:17,856 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 06:31:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.87it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.87it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:04<00:02, 13.80it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 21.45it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 21.45it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 21.45it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 21.45it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 21.45it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 21.45it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:01, 21.45it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:01, 21.45it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:01, 21.45it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:01, 21.45it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 29.83it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 29.83it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 29.83it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 29.83it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 29.83it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:04<00:00, 29.83it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:04<00:00, 29.83it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:04<00:00, 29.83it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:04<00:00, 29.83it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:04<00:00, 29.83it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:04<00:00, 29.83it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 40.46it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 40.46it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 40.46it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 40.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.30it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 24.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 24.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 24.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 24.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.99it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.99it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.71 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.99it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.70it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.70it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.70it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.70it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.70it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.70it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.76it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.76it/s]

    Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.76it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.76it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.76it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.76it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 39.95it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.95it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=288 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 42.74it/s]

    Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=224 avail_mem=76.66 GB):  60%|██████    | 35/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  60%|██████    | 35/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.67it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.67it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.67it/s]Capturing num tokens (num_tokens=160 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.67it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.67it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.67it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.50it/s]

    Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.50it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=64 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.00it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.00it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.00it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.00it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.00it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.00it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.73it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.73it/s]

    Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.73it/s] Capturing num tokens (num_tokens=4 avail_mem=76.60 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.73it/s]Capturing num tokens (num_tokens=4 avail_mem=76.60 GB): 100%|██████████| 58/58 [00:01<00:00, 37.69it/s]


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
    Generated text:  Karina and I’m a PhD student in the Division of Biology at the University of California, Berkeley. I am interested in understanding the molecular and cellular mechanisms of evolutionary adaptation. I have a particular interest in the role of genes, particularly transcription factors, in the regulation of gene expression in response to environmental cues, and their function in the control of plant development, including flowering, senescence, and seed germination.
    My main research goal is to use molecular genetic tools to investigate how these transcription factors interact and cooperate to control the processes of plant development and life cycle. We will use small RNA-based tools to reveal the role of these transcription
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to persuade an audience of 300 people at a concert to vote in favor of a bill that would require the government to issue a certain amount of a specific type of paper at a rate of $1 per unit. The bill was initially rejected by 100 people and is now being debated. If the bill is to be approved, at least 250 people must support it. If there is a 30% chance of a 5% refund for each person who opposes the bill, and the refund rate is 2% of the total bill amount, what is the probability that the bill will be
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a city of more than 2 million people, and it is the largest city in the world by population. Paris is a city of modernity, history, art, fashion, and gastronomy. Paris is one of the four major metropolitan areas in France, and it is also one of the four major metropolitan areas in Europe.
    
    It is a city that has a history of 700 years, and it was founded by the Romans. Paris was a key center of the Roman Empire. After its conquest by the French king Philip II, Paris became the capital of the French Kingdom. After the Fall of the French
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but scientists have made significant progress towards a more realistic understanding of the nature of intelligence. A professor at a prestigious university has been conducting experiments to study the effects of artificial intelligence on human behavior. Here are the steps he has taken:
    
    1. He observed a group of subjects who were trained to play a simple game of tennis.
    2. He programmed the AI to play tennis against the subjects.
    3. He measured the subjects' performance on a scale from 0 to 100.
    4. He recorded the subjects' responses to various stimuli, such as different types of tennis balls.
    5. He analyzed the data to identify


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


    Generated text:  [Name] and I am a [occupation] with [number of years] years of experience in [occupation]. I am a [occupation] with [number of years] years of experience in [occupation]. I am a [occupation] with [number of years] years of experience in [occupation]. I am a [occupation] with [number of years] years of experience in [occupation]. I am a [occupation] with [number of years] years of experience in [occupation]. I am a [occupation] with [number of years] years of experience in [occupation]. I am a [occupation] with [number of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French language. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is also home to many renowned French artists, writers, and musicians. Paris is a city of contrasts, with its modern
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, AI is likely to continue to be used for tasks such as fraud detection, cybersecurity, and environmental monitoring, as well as for tasks such as language translation and language generation. As AI becomes more integrated into our daily lives, we can expect to see a greater emphasis on ethical considerations and the development of responsible AI. Finally, AI is likely to continue to be used for tasks
    


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
    Generated text:  [Name], and I am a [Job Title] at [Company Name]. I enjoy [One or two hobbies, interests or activities] and I like to travel and enjoy different cultures. I am always looking for new experiences and ideas to improve my skills and knowledge. I am passionate about [What is your favorite hobby or activity]. What brings you to this job at [Company Name] and what does your day look like? Hello, my name is [Name] and I am a [Job Title] at [Company Name]. I enjoy [One or two hobbies, interests or activities] and I like to travel and enjoy different cultures
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known as the city of love and history. It is a bustling metropolis with a rich cultural heritage that has been shaped by its history and influences from various cultures. Paris is famous for its iconic landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. The city is also known for its delicious French cuisine and its rich history of culture and art. It is a must-visit destination for anyone interested in learning about French history and culture. Paris is a vibrant and exciting metropolis that has a rich history and a sense of community that makes it a truly unforgettable destination. 
    
    (Note
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and dynamic. Here are some possible trends that are currently being explored and are likely to emerge in the next decade:
    
    1. Increased focus on ethical AI: As AI becomes more integrated into society, it's likely that ethical concerns will become more important. This includes issues such as bias, privacy, and transparency. As a result, AI developers will be more focused on creating ethical AI that is built to work for the benefit of society as a whole.
    
    2. AI becomes more natural and human-like: As AI continues to evolve, it's likely that it will become more natural and human-like, allowing for more natural interactions with humans


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

     am

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ]

     [

    Occup

    ation

    ].

     I

    'm

     currently

     [

    Occup

    ation

    ]

     and

     I

     enjoy

     [

    Favorite

     Activity

     or

     Hobby

    ].

     
    


    I

     recently

     moved

     to

     [

    City

    /

    State

    ],

     and

     I

    've

     been

     living

     here

     for

     about

     [

    Years

    ].

     I

    'm

     [

    Favorite

     Color

    ], and

     I

     have

     a

     [

    Favorite

     Musical

     Instrument

    ]

     that

     I

     love

     to

     play

    .

     I

     believe

     in

     [

    Positive

     Aff

    irm

    ation

    ].

     
    


    I

     have

     a

     lot

     of

     [

    Favorite

     Book

     or

     Music

    ],

     and

     I

     like

     to

     [

    Favorite

     Activity

    ].

     I

     have

     a

     [

    Favorite

     Movie

     or

     TV

     Show

    ],

     and

     I

     enjoy

     [

    Favorite

     Food

    ].

     I

     have

     a

     [

    Favorite

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     also

     known

     as

     the

     "

    City

     of

     Light

    "

     and

     "

    City

     of

     Culture

    ".

     It

     is

     a

     cosm

    opolitan

     city

     with

     a

     rich

     history

     dating

     back

     to

     the

     

    7

    th

     century

    ,

     and

     is

     a

     popular

     tourist

     destination

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

     known

     for

     its

     vibrant

     arts

     scene

     and

     numerous

     festivals

     throughout

     the

     year

    .

     The

     city

     is

     a

     significant

     economic

     center

    ,

     hosting

     major

     businesses

     and

     industries

    ,

     including

     finance

    ,

     technology

    ,

     and

     tourism

    .

     Overall

    ,

     Paris

     is

     a

     fascinating

     city

     with

     a

     rich

     history

    ,

     diverse

     culture

    ,

     and

     a

     thriving

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     quite

     exciting

     and

     promising

    ,

     and

     it

    's

     difficult

     to

     predict

     exactly

     what

     the

     next

     few

     decades

     will

     bring

    .

     However

    ,

     here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     AI

     Accessibility

    :

     One

     of

     the

     most

     promising

     future

     trends

     is

     the

     increasing

     accessibility

     of

     AI

    .

     As

     AI

     technology

     becomes

     more

     advanced

    ,

     it

     will

     become

     more

     accessible

     to

     a

     broader

     range

     of

     people

    ,

     including

     those

     with

     disabilities

    ,

     who

     may

     not

     have

     had

     access

     to

     traditional

     forms

     of

     AI

     technology

     in

     the

     past

    .
    


    2

    .

     AI

     Ethics

     and

     Responsibility

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     lives

    ,

     there

     will

     be

     a

     growing

     focus

     on

     the

     ethics

     and

     responsibility

     of

     AI

     systems

    .

    



```python
llm.shutdown()
```
