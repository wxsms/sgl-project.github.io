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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.91it/s]


    2026-05-07 10:54:01,325 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 10:54:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.18it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.18it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.18it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.18it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  6.18it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  6.18it/s]

    Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:07,  6.18it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:07,  6.18it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 11.06it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 11.06it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 11.06it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 11.06it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 11.06it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 11.06it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 11.06it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 11.06it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 16.88it/s]

    Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 16.88it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:00, 24.38it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s]

    Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s]Compiling num tokens (num_tokens=12):  78%|███████▊  | 45/58 [00:05<00:00, 33.77it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 46.46it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 46.46it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 46.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=51.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=51.40 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=51.39 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=6656 avail_mem=51.39 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=6144 avail_mem=51.39 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=51.39 GB):   9%|▊         | 5/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=51.38 GB):   9%|▊         | 5/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=51.37 GB):   9%|▊         | 5/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=51.37 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.43it/s]Capturing num tokens (num_tokens=4608 avail_mem=51.37 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.43it/s]Capturing num tokens (num_tokens=4096 avail_mem=51.37 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=51.37 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.43it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=51.37 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=51.36 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=51.36 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=51.35 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=51.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=51.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.61it/s]Capturing num tokens (num_tokens=2560 avail_mem=51.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=51.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=51.34 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.61it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=51.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=51.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=51.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=51.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=51.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=51.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.84it/s]Capturing num tokens (num_tokens=960 avail_mem=51.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.84it/s] Capturing num tokens (num_tokens=896 avail_mem=51.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.84it/s]Capturing num tokens (num_tokens=832 avail_mem=51.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.84it/s]Capturing num tokens (num_tokens=768 avail_mem=51.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.84it/s]

    Capturing num tokens (num_tokens=768 avail_mem=51.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.49it/s]Capturing num tokens (num_tokens=704 avail_mem=51.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.49it/s]Capturing num tokens (num_tokens=640 avail_mem=51.31 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.49it/s]Capturing num tokens (num_tokens=576 avail_mem=51.31 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.49it/s]Capturing num tokens (num_tokens=512 avail_mem=51.30 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.49it/s]Capturing num tokens (num_tokens=512 avail_mem=51.30 GB):  50%|█████     | 29/58 [00:01<00:00, 30.19it/s]Capturing num tokens (num_tokens=480 avail_mem=51.31 GB):  50%|█████     | 29/58 [00:01<00:00, 30.19it/s]Capturing num tokens (num_tokens=448 avail_mem=51.31 GB):  50%|█████     | 29/58 [00:01<00:00, 30.19it/s]Capturing num tokens (num_tokens=416 avail_mem=51.31 GB):  50%|█████     | 29/58 [00:01<00:00, 30.19it/s]Capturing num tokens (num_tokens=384 avail_mem=51.31 GB):  50%|█████     | 29/58 [00:01<00:00, 30.19it/s]

    Capturing num tokens (num_tokens=352 avail_mem=51.30 GB):  50%|█████     | 29/58 [00:01<00:00, 30.19it/s]Capturing num tokens (num_tokens=352 avail_mem=51.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=320 avail_mem=51.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=288 avail_mem=51.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=256 avail_mem=51.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=240 avail_mem=51.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=224 avail_mem=51.28 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=224 avail_mem=51.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=208 avail_mem=51.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=192 avail_mem=51.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.11it/s]

    Capturing num tokens (num_tokens=176 avail_mem=51.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=160 avail_mem=51.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=144 avail_mem=51.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.11it/s]Capturing num tokens (num_tokens=144 avail_mem=51.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.28it/s]Capturing num tokens (num_tokens=128 avail_mem=51.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.28it/s]Capturing num tokens (num_tokens=112 avail_mem=51.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.28it/s]Capturing num tokens (num_tokens=96 avail_mem=51.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.28it/s] Capturing num tokens (num_tokens=80 avail_mem=51.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.28it/s]Capturing num tokens (num_tokens=64 avail_mem=51.25 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.28it/s]

    Capturing num tokens (num_tokens=64 avail_mem=51.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.48it/s]Capturing num tokens (num_tokens=48 avail_mem=51.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.48it/s]Capturing num tokens (num_tokens=32 avail_mem=51.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.48it/s]Capturing num tokens (num_tokens=28 avail_mem=51.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.48it/s]Capturing num tokens (num_tokens=24 avail_mem=51.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.48it/s]Capturing num tokens (num_tokens=24 avail_mem=51.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.01it/s]Capturing num tokens (num_tokens=20 avail_mem=51.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.01it/s]

    Capturing num tokens (num_tokens=16 avail_mem=51.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.01it/s]Capturing num tokens (num_tokens=12 avail_mem=51.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.01it/s]Capturing num tokens (num_tokens=8 avail_mem=51.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.01it/s] Capturing num tokens (num_tokens=8 avail_mem=51.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 28.01it/s]Capturing num tokens (num_tokens=4 avail_mem=51.22 GB):  98%|█████████▊| 57/58 [00:01<00:00, 28.01it/s]Capturing num tokens (num_tokens=4 avail_mem=51.22 GB): 100%|██████████| 58/58 [00:02<00:00, 28.95it/s]


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
    Generated text:  Stefan. I'm a member of the Scottish Royal Family. I was born in Dunfermline, Scotland and am a descendant of Queen Mary and King James VI. I have two brothers and two sisters. I went to school and worked as a music teacher before I became a doctor. I'm a writer, photographer and visual artist. My work is inspired by the way the human spirit can overcome adversity, whether it's a physical one, emotional, or spiritual. I am a regular columnist for the Guardian and a regular contributor for The Guardian's children's magazine. I'm currently the editor of The Monke and I'm planning on
    ===============================
    Prompt: The president of the United States is
    Generated text:  married to a person of the same age as the president, who is 46 years old. If the president has an odd number of children, each child has 3 siblings, and the president is the only child of the president's spouse. If the president has an even number of children, each child has 4 siblings, and the president is the only child of the president's spouse. How many years ago was the president born?
    To determine how many years ago the president was born, we need to analyze the information given step by step.
    
    1. Identify the current age of the president:
       - The president is 4
    ===============================
    Prompt: The capital of France is
    Generated text: ____. A. Paris B. London C. Moscow D. New York
    Answer:
    
    A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Moscow
    D. New York
    Answer:
    
    A
    
    The capital of France is ____
    A. Paris
    B. London
    C. Moscow
    D. New York
    Answer:
    
    A
    
    What is the capital of France? ( )
    A. Paris
    B. London
    C. Moscow
    D. New York
    Answer:
    
    A
    
    Which of the following is NOT a basic information element in the C3 system data?
    A. Organizational information
    ===============================
    Prompt: The future of AI is
    Generated text:  a rapidly evolving field with opportunities to transform our lives for the better. However, it’s not always clear which algorithms are best suited to solve specific problems.
    AI algorithms are different in many ways from traditional computer languages. The sheer number of variables can make it difficult to understand the complex algorithm behind them. At the same time, the amount of data that is collected and analyzed is growing at an unprecedented rate. As a result, it can be difficult to select the most effective algorithms for specific tasks.
    In this blog post, we’ll explore what makes a good algorithm and how it can be evaluated. We’ll also cover some of the key differences


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


    Generated text:  Paris, also known as the City of Light, a historic and cultural center with a rich history dating back to the Middle Ages. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe, as well as its diverse cuisine, fashion, and art scene. The city is also home to numerous museums, theaters, and other cultural institutions, making it a popular tourist destination. Paris is a vibrant and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI, there will be a greater emphasis on ethical AI. This will likely lead to more rigorous testing and validation of AI systems, as well as greater transparency and accountability in their development and deployment.
    
    2. Integration of AI with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This will allow for more complex and sophisticated
    


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
    Generated text:  [Name], and I'm an AI that specializes in answering questions and providing information on various topics. I'm always ready to help people find the answers they're looking for. How can I assist you today? [Name] looks and sounds like someone who is excited to learn and share their knowledge. They have a friendly and helpful demeanor and are always eager to answer any questions or concerns they have. What can I do for you today? [Name] is a bit of a puzzle, but I'm here to help. I love engaging with people and finding new ways to assist them with their questions. What can I do for you?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its iconic Notre-Dame Cathedral, bustling street life, and extensive museums and art collections. While the city offers a wide range of entertainment and dining options, Paris is also famous for its opulent opulence, especially in the historic districts of the Left Bank. With its medieval heritage and eclectic mix of architectural styles, Paris is a city that is synonymous with luxury and creativity. The city is a cosmopolitan hub of culture, history, and art, and is considered one of the most important cities in the world. 
    
    Therefore, Paris is the largest city in France, with a population of over 2.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting, and it is expected to continue evolving and transforming in many ways. Here are some possible future trends in artificial intelligence:
    
    1. Increased automation and AI integration: With the rise of AI, we are likely to see an increase in automation and the integration of AI into various industries. This could lead to increased efficiency, cost savings, and better outcomes for businesses and individuals.
    
    2. AI in healthcare: AI has the potential to revolutionize the healthcare industry by improving diagnostics, drug discovery, and patient care. AI could help doctors make more accurate diagnoses, develop new treatments, and predict the outcomes of medical procedures.
    
    3. AI in education


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

    profession

    ,

     age

    ,

     etc

    .]

     [

    Occup

    ation

    ].

     I

     have

     been

     [

    career

     goals

    ]

     for

     [

    number

     of

     years

    ]

     and

     [

    goals

    ].

     What

     brings

     you

     here

     today

    ?
    


    I

     am

     excited

     to

     meet

     you

     and

     start

     our

     conversation

    .

     What

     can

     I

     do

     for

     you

     today

    ?
    


    This

     self

    -int

    roduction

     was

     written

     in

     the

     style

     of

     a

     fictional

     character

    ,

     and

     I

     will

     be

     speaking

     with

     a

     character

     named

     [

    Name

    ]

     in

     a

     fictional

     book

    .

     The

     character

    's

     name

     is

     [

    Name

    ]

     and

     they

     are

     a

     [

    profession

    ,

     age

    ,

     etc

    .]

     [

    Occup

    ation

    ].

     They

     have

     been

     [

    career

     goals

    ]

     for

     [

    number

     of

     years

    ]

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    What

     is

     the

     capital

     of

     India

    ?

     The

     capital

     of

     India

     is

     New

     Delhi

    .

     How

     about

     the

     capital

     of

     Brazil

    ?

     The

     capital

     of

     Brazil

     is

     Bras

    ilia

    .
    


    Considering

     these

     facts

    ,

     what

     is

     the

     capital

     of

     a

     country

     that

     has

     

    1

    0

    0

    %

     Muslim

     population

    ?

     The

     capital

     of

     a

     country

     with

     

    1

    0

    0

    %

     Muslim

     population

     is

     Islamabad

    .

     How

     about

     a

     country

     with

     a

     minority

     population

    ?

     The

     capital

     of

     a

     country

     with

     a

     minority

     population

     is

     Washington

    ,

     D

    .

     C

    .
    


    What

     is

     the

     capital

     of

     Japan

    ?

     The

     capital

     of

     Japan

     is

     Tokyo

    .

     How

     about

     a

     country

     with

     a

     minority

     population

    ?

     The

     capital

     of

     a

     country

     with

     a

     minority

     population

     is

     K

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     has

     the

     potential

     to

     change

     the

     world

     in

     many

     ways

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     With

     the

     help

     of

     AI

    ,

     it

     is

     possible

     to

     provide

     more

     personalized

     experiences

     to

     users

    .

     AI

     can

     analyze

     user

     behavior

     and

     preferences

     to

     offer

     tailored

     recommendations

     and

     recommendations

    .
    


    2

    .

     Autonomous

     and

     Autonomous

     Vehicles

    :

     Autonomous

     cars

     and

     self

    -driving

     trucks

     are

     already

     becoming

     more

     prevalent

     in

     the

     industry

    .

     With

     the

     advancement

     of

     AI

    ,

     it

     is

     likely

     that

     we

     will

     see

     even

     more

     autonomous

     vehicles

     on

     the

     road

     in

     the

     future

    .
    


    3

    .

     Artificial

     Intelligence

     in

     Healthcare

    :

     AI

     is

     already

     being

     used

     to

     diagnose

     and

     treat

     diseases

    ,

     and

     it

     has

     the

     potential

    



```python
llm.shutdown()
```
