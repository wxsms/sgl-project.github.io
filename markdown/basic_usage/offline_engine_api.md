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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.68it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.67it/s]


    2026-05-11 06:50:19,074 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 06:50:19] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.69it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.69it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 22.69it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 22.69it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 22.69it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 22.69it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 22.69it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 22.69it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 22.69it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 22.69it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 30.81it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  21%|██        | 12/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  21%|██        | 12/58 [00:00<00:01, 28.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.54it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.54it/s] Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.89it/s]Capturing num tokens (num_tokens=896 avail_mem=74.09 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.89it/s]Capturing num tokens (num_tokens=832 avail_mem=74.08 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.89it/s]Capturing num tokens (num_tokens=768 avail_mem=74.08 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.89it/s]Capturing num tokens (num_tokens=704 avail_mem=74.08 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.89it/s]

    Capturing num tokens (num_tokens=704 avail_mem=74.08 GB):  45%|████▍     | 26/58 [00:00<00:01, 29.80it/s]Capturing num tokens (num_tokens=640 avail_mem=74.07 GB):  45%|████▍     | 26/58 [00:00<00:01, 29.80it/s]Capturing num tokens (num_tokens=576 avail_mem=74.06 GB):  45%|████▍     | 26/58 [00:00<00:01, 29.80it/s]Capturing num tokens (num_tokens=512 avail_mem=73.57 GB):  45%|████▍     | 26/58 [00:00<00:01, 29.80it/s]Capturing num tokens (num_tokens=480 avail_mem=73.58 GB):  45%|████▍     | 26/58 [00:00<00:01, 29.80it/s]Capturing num tokens (num_tokens=480 avail_mem=73.58 GB):  52%|█████▏    | 30/58 [00:00<00:00, 31.53it/s]Capturing num tokens (num_tokens=448 avail_mem=73.49 GB):  52%|█████▏    | 30/58 [00:00<00:00, 31.53it/s]Capturing num tokens (num_tokens=416 avail_mem=73.42 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.53it/s]Capturing num tokens (num_tokens=384 avail_mem=73.41 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.53it/s]Capturing num tokens (num_tokens=352 avail_mem=73.41 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.53it/s]

    Capturing num tokens (num_tokens=320 avail_mem=73.40 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.53it/s]Capturing num tokens (num_tokens=320 avail_mem=73.40 GB):  60%|██████    | 35/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=288 avail_mem=73.40 GB):  60%|██████    | 35/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=256 avail_mem=73.40 GB):  60%|██████    | 35/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=240 avail_mem=73.40 GB):  60%|██████    | 35/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=224 avail_mem=73.39 GB):  60%|██████    | 35/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=208 avail_mem=73.39 GB):  60%|██████    | 35/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=208 avail_mem=73.39 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=192 avail_mem=73.39 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=176 avail_mem=73.38 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=160 avail_mem=73.38 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.79it/s]

    Capturing num tokens (num_tokens=144 avail_mem=73.38 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=128 avail_mem=73.38 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=128 avail_mem=73.38 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=112 avail_mem=73.37 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=96 avail_mem=73.37 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.14it/s] Capturing num tokens (num_tokens=80 avail_mem=73.37 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=64 avail_mem=73.36 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=48 avail_mem=73.36 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.14it/s]Capturing num tokens (num_tokens=48 avail_mem=73.36 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=32 avail_mem=73.36 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=28 avail_mem=73.35 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.07it/s]

    Capturing num tokens (num_tokens=24 avail_mem=73.35 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=20 avail_mem=73.34 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=16 avail_mem=73.34 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.07it/s]Capturing num tokens (num_tokens=16 avail_mem=73.34 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.28it/s]Capturing num tokens (num_tokens=12 avail_mem=73.34 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.28it/s]Capturing num tokens (num_tokens=8 avail_mem=73.34 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.28it/s] Capturing num tokens (num_tokens=4 avail_mem=73.33 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.28it/s]Capturing num tokens (num_tokens=4 avail_mem=73.33 GB): 100%|██████████| 58/58 [00:01<00:00, 35.43it/s]


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
    Generated text:  Alex and I'm a 5th grade student. I'm really excited to learn new things and try new things. What are some fun activities you can do for me to help me learn?
    
    Learning new things and trying new things can be a fun and exciting experience! Here are some fun activities you can do with Alex to help him learn:
    
    1. **Reading**: Encourage Alex to read different books. This not only helps him learn new words but also exposes him to different topics and perspectives.
    
    2. **Writing**: Start a journal or an online blog. This can help him express his thoughts and ideas in a creative way.
    
    3
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. The American football coach is a person. Therefore, the American football coach is a person. 
    So, what does it follow that 
    (A) the president of the United States is a person. 
    (B) the American football coach is a person. 
    (C) the president of the United States is a person. 
    (D) the American football coach is a person. 
    (E) none of the above
    To determine the logical reasoning involved, we need to carefully analyze the given statements and see if they necessarily lead to any of the conclusions listed. Let's break it down step by step.
    
    1. **Given Statements:
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Which of the following statements about the population of Paris is correct? ① The proportion of people who like music is greater than that who like beauty. ② The proportion of people who like beauty is greater than that who like music. ③ The proportion of people who like music is less than that who like beauty. ④ The proportion of people who like beauty is less than that who like music. ⑤ The proportion of people who like music is greater than that who like beauty.
    A. ①④
    B. ②③
    C. ①③
    ===============================
    Prompt: The future of AI is
    Generated text:  already here. It is already changing the world in ways we’ve never seen. But before that future comes to fruition, you need to understand how it’s going to impact your business. Our team of expert AI experts can help you plan and execute a successful AI strategy.
    Why is AI so important?
    AI is transforming the way we live our lives. It is making life easier, more efficient and less expensive. We’re all living our lives better thanks to AI.
    In the 21st century, we live in a digital age. And digital technology has become an integral part of every aspect of life. In fact, it’s hard


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Expertise] who has been [Number of Years] years in the field of [Field of Interest]. I'm passionate about [Reason for Passion] and I'm always looking for ways to [Action or Goal]. I'm a [Skill or Expertise] who is always [Reason for Passion] and I'm always looking for ways to [Action or Goal]. I'm a [Skill or Expertise] who is always [Reason for Passion] and I'm always looking for ways to [Action or Goal]. I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its rich history, beautiful architecture, and vibrant culture, and is a popular tourist destination. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its fashion industry, with many famous fashion houses and designers operating in the city. The city is a major center of business and commerce, and is home to many important institutions such
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread adoption of AI in healthcare.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes and improve quality control. As AI technology continues to improve, we can expect to see even more widespread adoption of AI in manufacturing.
    
    3. AI in finance: AI
    


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
    Generated text:  [Name], and I'm a [occupation] with [number of years] years of experience in [specific field]. I have a diverse range of interests, which I enjoy sharing with others, and I value helping and supporting my community. I have a soft spot for animals and have a passion for nature conservation, and I'm always looking for ways to protect our planet and its inhabitants. I'm enthusiastic and love to learn new things, and I'm always open to new experiences. Thank you for having me! Let's connect and explore how we can work together to create a better future for all! [Name] [Age] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    It is a major European city, known for its historical significance, arts and culture, and culinary excellence. It has a population of over 1. 5 million people and is one of the most cosmopolitan cities in Europe. The city is situated on the Seine River, which flows through its heart. Paris is famous for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, and its cuisine is renowned worldwide. The city is also known for its fashion industry, particularly the couturier Louis Vuitton, and its unique art scene. Paris is a global destination for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by a number of trends and advancements that will continue to reshape the technology, society, and economy. Here are some of the potential future trends in artificial intelligence:
    
    1. Autonomous robots and drones: As AI continues to improve, we may see the development of autonomous robots and drones that can perform a variety of tasks, from construction to defense. These robots could potentially replace human workers in some areas, but could also have positive environmental and ethical impacts.
    
    2. Biometric recognition: As AI continues to improve, we may see the development of more advanced biometric recognition technology, such as facial recognition and voice recognition. This could


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

    Your

     Name

    ]

     and

     I

     am

     a

     [

    Your

     Profession

     or

     Area

     of

     Expert

    ise

    ]

     with

     [

    Your

     Age

    ]

     years

     of

     experience

     in

     [

    Your

     Area

     of

     Expert

    ise

    ].

     I

     am

     a

     passionate

     [

    Your

     Passion

    ],

     and

     I

     am

     always

     learning

     new

     things

    ,

     and

     I

     am

     always

     striving

     to

     improve

     myself

    .

     What

     is

     your

     profession

     or

     field

     of

     expertise

    ?

     What

     are

     your

     hobbies

     or

     interests

     outside

     of

     work

    ?

     What

     are

     your

     most

     memorable

     moments

     or

     experiences

    ?

     How

     do

     you

     maintain

     a

     balance

     between

     work

     and

     personal

     life

    ?


    The

     characters

     in

     this

     scenario

     are

     humans

    ,

     as

     this

     is

     a

     fictional

     setting

    .

     The

     introduction

     should

     be

     neutral

     and

     formal

    ,

     without

     personal

     bias

     or

     judgment

    .

     It

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     answer

     is

    :

     Paris

     is

     the

     capital

     city

     of

     France

    .

     
    


    This

     is

     a

     factual

     statement

     that

     can

     be

     succinct

    ly

     summarized

     as

    :

     "

    Paris

     is

     the

     capital

     of

     France

    ."

     This

     statement

     encaps

    ulates

     the

     key

     information

     provided

     in

     the

     original

     question

     and

     provides

     a

     clear

    ,

     concise

     answer

     to

     the

     question

    .

     It

     is

     a

     straightforward

     and

     easy

    -to

    -under

    stand

     statement

     about

     the

     city

    's

     status

     as

     its

     country

    's

     capital

    .

     
    


    To

     elaborate

     further

    :


    -

     "

    The

     capital

    "

     refers

     to

     the

     seat

     of

     the

     government

    ,

     legislative

     body

    ,

     and

     political

     center

     of

     a

     country

    .


    -

     In

     the

     case

     of

     France

    ,

     Paris

     is

     both

     the

     seat

     of

     the

     French

     government

     and

     the

     capital

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     wide

     range

     of

     developments

     and

     changes

    ,

     as

     technology

     continues

     to

     evolve

    .

     Here

     are

     some

     of

     the

     most

     likely

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

     increasingly

     integrated

     with

     other

     technologies

     such

     as

     machine

     learning

    ,

     cloud

     computing

    ,

     and

    物联网

    （

    物联网

    ，

    中文

    译

    为

    物联网

    ）

    .

     This

     integration

     will

     enable

     AI

     to

     perform

     more

     complex

     tasks

     and

     improve

     its

     ability

     to

     understand

     and

     respond

     to

     new

     types

     of

     inputs

     and

     interactions

    .
    


    2

    .

     Automation

     of

     mundane

     tasks

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     likely

     that

     it

     will

     be

     able

     to

     perform

     many

     tasks

     that

     are

     currently

     done

     by

     humans

    .

     For

     example

    ,

     AI

    



```python
llm.shutdown()
```
