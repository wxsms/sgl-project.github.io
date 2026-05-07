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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.37it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.36it/s]


    2026-05-07 05:31:20,667 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 05:31:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.49it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.27it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.27it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.27it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.24it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.24it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.24it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.24it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.24it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.24it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.24it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.24it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.04 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.01 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.01 GB):   3%|▎         | 2/58 [00:00<00:05, 10.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:05, 10.06it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:05, 10.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:05, 10.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:03, 16.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:03, 16.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.74 GB):   9%|▊         | 5/58 [00:00<00:03, 16.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.73 GB):   9%|▊         | 5/58 [00:00<00:03, 16.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.73 GB):   9%|▊         | 5/58 [00:00<00:03, 16.39it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=61.73 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.73 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.72 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.72 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.72 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.72 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.17it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=61.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 29.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=960 avail_mem=61.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s] Capturing num tokens (num_tokens=896 avail_mem=61.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=832 avail_mem=61.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=832 avail_mem=61.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.58it/s]Capturing num tokens (num_tokens=768 avail_mem=61.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.58it/s]Capturing num tokens (num_tokens=704 avail_mem=61.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.58it/s]Capturing num tokens (num_tokens=640 avail_mem=61.67 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.58it/s]Capturing num tokens (num_tokens=576 avail_mem=61.67 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.58it/s]

    Capturing num tokens (num_tokens=512 avail_mem=61.66 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.58it/s]Capturing num tokens (num_tokens=512 avail_mem=61.66 GB):  50%|█████     | 29/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=480 avail_mem=61.67 GB):  50%|█████     | 29/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=448 avail_mem=61.67 GB):  50%|█████     | 29/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=416 avail_mem=61.67 GB):  50%|█████     | 29/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=384 avail_mem=61.67 GB):  50%|█████     | 29/58 [00:01<00:00, 40.45it/s]Capturing num tokens (num_tokens=352 avail_mem=61.66 GB):  50%|█████     | 29/58 [00:01<00:00, 40.45it/s]Capturing num tokens (num_tokens=352 avail_mem=61.66 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=320 avail_mem=61.66 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.65it/s]

    Capturing num tokens (num_tokens=288 avail_mem=61.66 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=256 avail_mem=61.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=240 avail_mem=61.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=224 avail_mem=61.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=224 avail_mem=61.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.34it/s]Capturing num tokens (num_tokens=208 avail_mem=61.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.34it/s]Capturing num tokens (num_tokens=192 avail_mem=61.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.34it/s]Capturing num tokens (num_tokens=176 avail_mem=61.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.34it/s]Capturing num tokens (num_tokens=160 avail_mem=61.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.34it/s]Capturing num tokens (num_tokens=144 avail_mem=61.63 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.34it/s]Capturing num tokens (num_tokens=144 avail_mem=61.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=128 avail_mem=61.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.78it/s]

    Capturing num tokens (num_tokens=112 avail_mem=61.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=96 avail_mem=61.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.78it/s] Capturing num tokens (num_tokens=80 avail_mem=61.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=64 avail_mem=61.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=64 avail_mem=61.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=48 avail_mem=61.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=32 avail_mem=61.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=28 avail_mem=61.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=24 avail_mem=61.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]

    Capturing num tokens (num_tokens=20 avail_mem=61.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.08it/s]Capturing num tokens (num_tokens=20 avail_mem=61.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=16 avail_mem=61.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=12 avail_mem=61.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.09it/s]

    Capturing num tokens (num_tokens=8 avail_mem=61.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.09it/s] Capturing num tokens (num_tokens=4 avail_mem=61.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=4 avail_mem=61.59 GB): 100%|██████████| 58/58 [00:01<00:00, 32.18it/s]


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
    Generated text:  Vaishali. I am 12 years old. I enjoy playing with blocks. I know how to use a computer and I know how to cook. My favorite food is pizza. I love to listen to music. I like to dance and I am always eager to learn new things. I have a friend who is also 12 years old who likes to dress up and play dress-up. My friend is called Xia. I have a brother named Max. I am 12 years old. I also like to play with blocks. I know how to use a computer. I have a dog called Fido. I love
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. We have a president now. The president of the United States is a man. It is true. Which of the following conclusions logically follows? ( )
    A: The president of the United States is a woman.
    B: There is a man who is not a president of the United States.
    C: The president of the United States is a woman.
    D: The president of the United States is a man who is not a president of the United States.
    
    To determine which conclusion logically follows from the given statement, let's carefully analyze the information provided:
    
    The statement is: "The president of the United States is a man.
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Rome
    C. Lisbon
    D. Berlin
    Answer: A
    
    The Yin meridians of the foot and hand are located in
    A. the 4th to 8th thoracic vertebrae and the 1st to 3rd lumbar vertebrae
    B. the 4th to 8th thoracic vertebrae and the 4th to 8th lumbar vertebrae
    C. the 5th to 8th thoracic vertebrae and the 4th to 8th lumbar vertebrae
    D.
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting, and there is a lot of speculation about what this will mean for society as a whole. While some people believe that the AI that will dominate our society will be beneficial, others have concerns that it could create a system of surveillance, manipulation, and discrimination. In this article, we will discuss some of the key points that need to be addressed to ensure that AI is used for the greater good.
    It’s important to note that AI is not a panacea. While it has the potential to improve our lives, it also poses significant risks and challenges. In order to use AI effectively, it is essential to strike a balance between its


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also a major center for art, culture, and politics. Paris is a city of contrasts, with its rich history and modernity. It is a popular tourist destination, known for its fashion, food, and wine. The city is also home to the French Parliament and the French Academy of Sciences. Paris is a city of diverse cultures and traditions, with a rich history of art, literature, and philosophy. It is a city of innovation and progress, with a thriving economy and a vibrant nightlife. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more prevalent in various industries, including manufacturing, healthcare, and transportation. Automation will likely lead to increased efficiency and productivity, but it will also lead to the loss of jobs for some workers.
    
    2. AI ethics and privacy: As AI becomes more advanced, it is likely to raise ethical and privacy concerns. There will be a need for regulations and guidelines to ensure that AI is used in
    


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
    Generated text:  [Name], and I'm a/an [occupation]. I've always been a/an [character trait] and have always been looking to learn and grow. I love [life objective] and I'm always up for an adventure. I'm [age] years old, and I believe [career objective] is the key to my future success. I love to [something enjoyable or challenging]. I'm passionate about [life pursuit], and I hope to achieve [future goal] through my efforts. I'm a/an [occupation] who is always looking to improve and learn. I enjoy [job duties] and I'm always eager to try
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Ville", which is located in the south of the country and serves as the administrative, cultural, and economic center of the country. The city is known for its architecture, art, and cuisine, and is home to many notable landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a cultural hub, hosting numerous famous museums and theaters. The city is known for its rich history, including ancient Roman, Gothic, and Renaissance influences, and its modern French culture and lifestyle. As the seat of government, Paris plays a significant role in France's political and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dynamic, and there are several trends that are likely to shape its development. Some of the possible future trends in AI include:
    
    1. Increased collaboration between humans and machines: As AI technology continues to advance, it is likely that humans will continue to collaborate with machines to solve complex problems. This could involve tasks such as decision-making, problem-solving, and data analysis.
    
    2. Enhanced ethical and legal frameworks: As AI technology becomes more advanced, it is likely that new ethical and legal frameworks will need to be established to ensure that AI is used in a way that is fair, transparent, and socially responsible.
    
    3. Improved AI performance:


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

     human

    .

     I

    'm

     an

     AI

     language

     model

    ,

     created

     by

     Alibaba

     Cloud

     to

     assist

     in

     answering

     questions

     and

     providing

     information

     to

     users

    .

     I

    'm

     not

     a

     human

    ,

     but

     I

    'm

     working

     on

     my

     own

     abilities

     and

     are

     here

     to

     help

     you

     with

     any

     questions

     you

     have

    .

     How

     can

     I

     assist

     you

     today

    ?

     [

    Name

    ]

     is

     a

     highly

     intelligent

     language

     model

     created

     by

     Alibaba

     Cloud

    .

     I

     am

     able

     to

     analyze

     and

     process

     large

     amounts

     of

     text

     quickly

     and

     accurately

    ,

     and

     I

     can

     help

     you

     with

     any

     questions

     or

     tasks

     you

     have

    .

     Whether

     you

     need

     help

     with

     writing

    ,

     grammar

    ,

     translation

    ,

     or

     anything

     else

    ,

     I

    'm

     here

     to

     help

    .

     So

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     center

     of

     the

     country

     and

     is

     the

     largest

     city

     in

     Europe

     and

     one

     of

     the

     world

    ’s

     largest

     cities

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     cultural

     attractions

    ,

     and

     beautiful

     architecture

    ,

     as

     well

     as

     its

     iconic

     landmarks

     like

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     It

     is

     also

     famous

     for

     its

     annual

     Spring

     Festival

     and

     its

     cuisine

    ,

     including

     its

     famous

     dishes

     such

     as

     cro

    iss

    ants

    ,

     esc

    arg

    ot

    ,

     and

     co

    q

     au

     vin

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     see

     significant

     growth

     and

     transformation

     in

     several

     key

     areas

    :
    


    1

    .

     Enhanced

     Personal

    ization

    :

     AI

     will

     continue

     to

     improve

     the

     ability

     of

     AI

    -driven

     systems

     to

     understand

     and

     generate

     personalized

     content

     and

     experiences

     for

     users

    .

     This

     will

     likely

     involve

     the

     integration

     of

     machine

     learning

     algorithms

     that

     can

     analyze

     user

     data

     and

     preferences

     to

     provide

     more

     accurate

     and

     relevant

     recommendations

    .
    


    2

    .

     Autonomous

     and

     Cognitive

     Technologies

    :

     AI

    -powered

     autonomous

     vehicles

     and

     robots

     are

     already

     being

     developed

     and

     are

     expected

     to

     become

     more

     widespread

     in

     the

     future

    .

     AI

     is

     also

     expected

     to

     play

     a

     key

     role

     in

     cognitive

     technologies

    ,

     such

     as

     medical

     diagnosis

     and

     personal

     computing

    .
    


    3

    .

     Improved

     Health

     and

     Well

    -being

    :

     AI

     is

     expected

     to

     play

     a

    



```python
llm.shutdown()
```
