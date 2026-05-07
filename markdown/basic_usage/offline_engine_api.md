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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.25it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.25it/s]


    2026-05-07 23:18:35,793 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 23:18:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.49it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:04<00:01, 16.22it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 25.33it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 25.33it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 25.33it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 25.33it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 25.33it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 25.33it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 25.33it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 25.33it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 25.33it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:04<00:00, 25.33it/s]

    Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:04<00:00, 25.33it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 34.44it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 34.44it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 34.44it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 34.44it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 34.44it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 34.44it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 34.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.98it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.56it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.56it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.56it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.56it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.06it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.06it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.06it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.06it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.06it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.06it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 44.50it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  50%|█████     | 29/58 [00:00<00:00, 44.50it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 44.50it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 44.50it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 44.50it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 44.50it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 44.50it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=256 avail_mem=75.00 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=240 avail_mem=74.90 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=224 avail_mem=74.90 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=208 avail_mem=74.89 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=208 avail_mem=74.89 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.26it/s]Capturing num tokens (num_tokens=192 avail_mem=74.89 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.26it/s]Capturing num tokens (num_tokens=176 avail_mem=74.89 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=160 avail_mem=74.89 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=144 avail_mem=74.88 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.26it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.88 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=128 avail_mem=74.88 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=112 avail_mem=74.88 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=96 avail_mem=74.87 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s] Capturing num tokens (num_tokens=80 avail_mem=74.87 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=64 avail_mem=74.87 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=48 avail_mem=74.86 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=48 avail_mem=74.86 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=32 avail_mem=74.86 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=28 avail_mem=74.85 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=24 avail_mem=74.85 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=20 avail_mem=74.85 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.85 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=16 avail_mem=74.85 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.07it/s]Capturing num tokens (num_tokens=12 avail_mem=74.84 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.07it/s]Capturing num tokens (num_tokens=8 avail_mem=74.84 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.07it/s] Capturing num tokens (num_tokens=4 avail_mem=74.84 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.07it/s]Capturing num tokens (num_tokens=4 avail_mem=74.84 GB): 100%|██████████| 58/58 [00:01<00:00, 42.66it/s]


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
    Generated text:  Vicky. I'm a young woman and I'm 28 years old. I have a lot of hobbies like reading, watching TV, playing chess, reading books, writing, drawing, and so on. I like to think that I am the best person in the world because I'm a good person. My mother, who is my best friend, is always telling me to be a good person. She encourages me to be kind and considerate to everyone. I like to share my stories with my family and friends, and I hope that my stories will help them understand and appreciate what I have experienced. There is one thing I
    ===============================
    Prompt: The president of the United States is
    Generated text:  in the White House, and the President of the Philippines is visiting the White House. How many states are in the United States? To determine how many states are in the United States, we need to follow these steps:
    
    1. Identify the location of the president of the United States.
    2. Identify the location of the president of the Philippines.
    3. Determine which state the president of the United States is visiting.
    4. Count the number of states in the United States.
    
    From the problem, we know that the president of the United States is in the White House. The president of the Philippines is visiting the White House. Therefore, the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the third most populous city in the world, and the largest city in the European Union and the second-largest city in the European continent. Paris is the capital of the "ParisRegion" and has a capital of the "Cercle de Paris". It is situated in the northwestern part of France, on the bank of the Seine River. The capital has a territory of 665.5 km², and it is the second-largest in France by population.
    
    The year 2020 is the 20th year of existence of the "ParisRegion" and the 50th
    ===============================
    Prompt: The future of AI is
    Generated text:  in a pool of creative thinkers who are passionate about the future and want to help create a better future. The field of AI is changing rapidly, and there are many new techniques and approaches that are emerging, and new opportunities to explore.
    The field of AI has grown exponentially in the past decade, and with the growth of technology and the changing needs of the world, the field of AI is likely to continue to evolve and grow even more in the coming years. One of the areas of AI that is likely to continue to grow is the field of robotics, as robotics technology continues to evolve and become more complex.
    Robotics is a branch of


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and a vibrant culture. It is the largest city in France and the second-largest city in the European Union, with a population of over 10 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, with many famous fashion designers and boutiques located in the city. Paris is a cultural and artistic center, with many museums, theaters, and art galleries. It is also a major transportation hub
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more widespread use of AI in healthcare,
    


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
    Generated text: ... 
    
    You'll need to fill in the blank with the character's name to complete the self-introduction.
    
    John Doe, a junior at the University of California, Berkeley, is a passionate environmental activist. Despite his green thumb, John has a knack for turning his passion for sustainability into action. He volunteers at local eco-friendly organizations, teaches sustainability courses at the university, and encourages his friends to adopt eco-friendly lifestyles. He's also an avid reader, regularly turning the pages of environmental fiction and non-fiction books. John is a true advocate for a sustainable future and a wonderful person to have around. Who would you like to introduce yourself
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Parthenay" in its Latin name, and the city is located in the southern part of France, on the banks of the Seine river. It is the largest city in France and the second-largest city in the European Union after Brussels. Paris is famous for its art and culture, with many museums, theaters, and fashion landmarks. The city is also home to the French Parliament, the Louvre Museum, and the Eiffel Tower, which is the tallest building in the world. Paris is a cultural and economic center of France and has a large population of about 21 million people.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and depends on a variety of factors, including advances in technology, changing societal and regulatory environments, and the development of new technologies. Here are some potential trends in AI that could be seen in the years ahead:
    
    1. Increased use of AI in healthcare: As AI becomes more advanced and integrated into medical practices, there is a potential for it to revolutionize healthcare delivery. AI-powered diagnostic tools could lead to faster and more accurate diagnoses, reducing the time and cost associated with medical care. AI in drug discovery could also lead to faster and more effective drug development.
    
    2. Advancements in natural language processing (NLP): NLP


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

     name

    ],

     and

     I

    'm

     a

     [

    insert

     occupation

    ]

     with

     a

     passion

     for

     [

    insert

     hobby

     or

     interest

    ].

     I

    'm

     a

     [

    insert

     age

    ,

     gender

    ,

     and

     educational

     background

    ],

     and

     I

    'm

     currently

     [

    insert

     current

     job

     title

    ,

     role

    ,

     or

     any

     other

     relevant

     information

    ].

     I

    'm

     [

    insert

     personality

     traits

    ,

     such

     as

     being

     kind

    ,

     honest

    ,

     or

     creative

    ].

     And

     my

     [

    insert

     specific

     interest

    ,

     such

     as

     reading

    ,

     cooking

    ,

     or

     travel

    ]

     is

     [

    insert

     number

     or

     information

     about

     it

    ].

     Thank

     you

     for

     considering

     me

     for

     a

     short

     interview

    .

     I

    'm

     looking

     forward

     to

     hearing

     from

     you

    !

     [

    insert

     thank

     you

     message

    ].

     [

    insert

     your

     name

    ]

     [

    insert

     your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     where

     the

     E

    iff

    el

     Tower

     stands

    .

     Known

     for

     its

     iconic

     landmarks

    ,

     including

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

     the

     Ch

    amps

    -E

    lys

    ées

    ,

     it

    's

     a

     bustling

     met

    ropolis

     with

     a

     rich

     history

     dating

     back

     over

     

    1

    ,

    0

    0

    0

     years

    .

     The

     city

     is

     also

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

     and

     the

     Universal

     Exhibition

    ,

     among

     other

     attractions

    .

     Paris

     is

     a

     vibrant

     and

     culturally

     rich

     city

     that

     offers

     visitors

     a

     diverse

     range

     of

     experiences

    .

     It

    's

     also

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

     (

    This

     statement

     is

     based

     on

     historical

     fact

     and

     current

     attractions

    .

     However

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     complex

     and

     rapidly

     evolving

     area

    ,

     and

     there

     is

     no

     one

     "

    right

    "

     answer

    .

     However

    ,

     here

     are

     some

     potential

     trends

     that

     are

     likely

     to

     shape

     the

     AI

     landscape

     in

     the

     years

     to

     come

    :
    


    1

    .

     Increased

     Automation

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     they

     will

     be

     able

     to

     perform

     a

     wider

     range

     of

     tasks

    ,

     from

     data

     analysis

     to

     manual

     labor

    .

     This

     will

     lead

     to

     the

     automation

     of

     many

     jobs

    ,

     freeing

     up

     workers

     to

     focus

     on

     higher

    -value

     tasks

    .
    


    2

    .

     Integration

     of

     AI

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     be

     integrated

     into

     other

     technologies

    ,

     such

     as

     machine

     learning

    ,

     computer

     vision

    ,

     natural

     language

     processing

    ,

     and

     robotics

    .

     This

     integration

     will

     allow

     AI

    



```python
llm.shutdown()
```
