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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.38it/s]


    2026-04-29 20:07:51,948 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 20:07:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:00,  5.26s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:00,  5.26s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:00,  5.26s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:00,  5.26s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:00,  5.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.72it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  7.96it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  7.96it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  7.96it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  7.96it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  7.96it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  7.96it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  7.96it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  7.96it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:05<00:02, 11.68it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 18.17it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 18.17it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 18.17it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 18.17it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 18.17it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 18.17it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 18.17it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 18.17it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 18.17it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 18.17it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:05<00:01, 18.17it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 26.64it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 26.64it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 26.64it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 26.64it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 26.64it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 26.64it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 26.64it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:06<00:00, 26.64it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:06<00:00, 26.64it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:06<00:00, 26.64it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:06<00:00, 26.64it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:06<00:00, 26.64it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00, 37.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:04, 13.72it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:04, 13.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:04, 13.72it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:04, 13.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.94it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.94it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.94it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.94it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.10it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.10it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.10it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.35it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.35it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.35it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.35it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.35it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.35it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.11it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.11it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.11it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.11it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.11it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.26it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.26it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.26it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.26it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.26it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.26it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 40.57it/s]

    Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.25it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.25it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.25it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.25it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.25it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.25it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.50it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.50it/s]

    Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.68it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.68it/s]

    Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 37.82it/s]


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
    Generated text:  AbraZa and I'm a candidate for Transplantation of a brain.
    I had a long history of heart trouble.
    Right now my brain is not moving. I am in the middle of my second week of hospitalization. I am getting dizzy, weak, my hands and feet go numb and my leg is unresponsive. What do I do?
    
    Image result for abraza brain transplant
    
      1. I don’t want to try to transplant my brain either.
      2. I am not willing to donate my blood.
    
    I know transplant rejection is a possible outcome of a brain transplant, but what do I do
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the cabinet, and the cabinet is made up of the White House staff and five members of Congress. If each member of the cabinet has 50 employees and the president also has 50 employees, how many total employees are in the cabinet?
    To determine the total number of employees in the cabinet, we need to consider both the members of the cabinet and the president. Let's break it down step by step.
    
    1. Identify the number of members in the cabinet. The cabinet is made up of the White House staff and five members of Congress. Therefore, the total number of cabinet members is:
       \[
      
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Marseille
    C. London
    D. Berlin
    Answer:
    
    Paris
    
    Which of the following does not belong to the 'Four Confucian classics' of China?
    A. The Analects
    B. The Book of Changes
    C. The Mencius
    D. The Doctrine of the Mean
    Answer:
    
    D
    
    The capital of France is ____
    A. Paris
    B. Marseille
    C. London
    D. Berlin
    Answer:
    
    Paris
    
    A city's administrative division includes ____.
    A. County
    B. Town
    C. City
    D. Municipality
    Answer:
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  very uncertain. AI is growing in popularity, but it’s not as popular as it could be. AI is being used in many different ways. It can be used in healthcare, financial services, transportation, and many other areas. There are many different ways that AI is being used, but some of the most common uses are recognizing faces, predicting behavior, and identifying patterns in data. AI is also being used to develop new technologies and products, and it’s being used to improve our lives in many other ways.
    As the technology behind AI continues to improve, it is becoming more and more valuable in different areas of our lives. AI has


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


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, as well as its vibrant arts scene and culinary delights. Paris is a cultural and political center of France and a major tourist destination, attracting millions of visitors each year. The city is also home to many important institutions such as the French Academy of Sciences and the French Parliament. Overall, Paris is a city of contrasts and beauty that has made it a global
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the potential future trends in AI:
    
    1. Increased automation and artificial intelligence: As automation and AI continue to advance, we are likely to see more and more jobs automated, which could lead to a shift in the job market. However, this could also create new opportunities for people to work in areas such as data analysis, machine learning, and software development.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there will be an increased need for security measures to protect against cyber
    


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
    Generated text:  [Name] and I am a [role] in [Company] who have been working here for [number of years].
    I am a [type of person] with [at least one notable skill or achievement]. I am here because I am interested in [reason for being here], and I am committed to [why I am here]. I am excited to contribute to [reason for being here] and I am always eager to learn and grow as a professional.
    If you have any questions or concerns, feel free to reach out to me. I am always here to assist you. [Name] is looking forward to meeting you! [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    France's capital city is Paris. It is the largest city in Europe and is home to many of the country's major cultural and historical attractions, including the Louvre Museum, the Arc de Triomphe, and the Eiffel Tower. Paris is also known for its vibrant nightlife and unique culture, with its famous museums and cafes. The French capital has a rich history dating back to ancient times, and it is a bustling and dynamic city with a strong sense of identity and tradition. The city is home to over 10 million people and is often referred to as the "city of love" due to its lively nightlife
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a variety of potential trends, including the development of more powerful, faster, and more capable AI systems. Additionally, there is a growing emphasis on ethical considerations and the development of AI that is aligned with human values and principles. AI systems may also become more capable of learning and adapting to new situations, and may be able to interact with humans in a more natural and interactive way. Finally, there is a potential for AI to be used in a variety of applications beyond just general-purpose computing, such as healthcare, transportation, and energy, and could have a significant impact on society. 
    
    Some of the key areas of


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

    Age

    ]

     year

     old

     [

    Gender

    ]

     [

    Prof

    ession

    ].

     I

    'm

     passionate

     about

     [

    Your

     Profession

     or

     Hobby

    ],

     and

     I

     enjoy

     [

    Some

     Past

    ime

     or

     Activity

    ],

     such

     as

     [

    Specific

     Activity

    ].

     I

     believe

     in

     [

    Personal

     Values

     or

     Eth

    ical

     Principles

    ].

     I

    'm

     confident

     in

     [

    Strength

    s

     or

     Abilities

    ],

     and

     I

     enjoy

     [

    H

    obbies

     or

     Inter

    ests

    ].

     I

    'm

     looking

     forward

     to

     [

    Future

     Goals

     or

     Adv

    ancement

    ].

     I

    'm

     excited

     to

     [

    Join

     or

     Start

     a

     New

     Project

     or

     Team

    ].

     I

     believe

     in

     [

    M

    oral

     Values

     or

     Eth

    ical

     Con

    ventions

    ].

     I

    'm

     dedicated

     to

     [

    Personal

     Ende

    avors

     or

     Projects

    ].

     I

    'm

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     as

     the

     City

     of

     Light

     due

     to

     its

     rich

     history

     and

     vibrant

     culture

    .

     
    


    In

     this

     city

    ,

     one

     can

     find

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

     the

     Arc

     de

     Tri

    omp

    he

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     more

     iconic

     landmarks

    .

     Paris

     is

     also

     famous

     for

     its

     cuisine

    ,

     including

     cro

    iss

    ants

    ,

     past

    ries

    ,

     and

     gourmet

     wine

    ,

     and

     its

     fashion

     scene

    ,

     featuring

     iconic

     designers

     like

     Louis

     V

    uit

    ton

     and

     Chanel

    .

     Additionally

    ,

     Paris

     is

     home

     to

     many

     of

     the

     world

    's

     cultural

     institutions

    ,

     including

     the

     Metropolitan

     Museum

     of

     Art

     and

     the

     Lou

    vre

     Museum

    .

     
    


    Despite

     its

     size

     and

     population

    ,

     Paris

     is

     a

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     continued

     advancements

     in

     several

     areas

    ,

     including

    :
    


    1

    .

     Increased

     accuracy

    :

     AI

     systems

     are

     getting

     more

     accurate

     at

     recognizing

     patterns

     and

     making

     predictions

    .

     This

     is

     due

     to

     advances

     in

     machine

     learning

     algorithms

    ,

     which

     allow

     machines

     to

     learn

     from

     more

     data

     and

     make

     more

     accurate

     predictions

    .
    


    2

    .

     Enhanced

     creativity

    :

     AI

     is

     already

     capable

     of

     generating

     new

     ideas

     and

     creative

     outputs

    ,

     such

     as

     artistic

     or

     scientific

     work

    .

     Future

     AI

     systems

     may

     be

     designed

     to

     enhance

     creativity

     by

     allowing

     them

     to

     process

     and

     generate

     more

     complex

     ideas

    .
    


    3

    .

     Personal

    ization

    :

     AI

     is

     already

     being

     used

     to

     personalize

     experiences

     for

     users

    ,

     such

     as

     through

     email

     marketing

     or

     website

     content

    .

     Future

     AI

     systems

     may

     be

     designed

    



```python
llm.shutdown()
```
