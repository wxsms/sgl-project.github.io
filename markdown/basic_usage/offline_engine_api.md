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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.15it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.15it/s]


    2026-05-07 05:52:39,216 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 05:52:39] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.49it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.17it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.17it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.17it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.17it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.17it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.17it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.17it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.17it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.17it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.17it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.87it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.87it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.87it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.87it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:04, 12.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:04, 12.65it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:04, 12.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:04, 12.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:03, 16.69it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.79it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.33it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.73it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.73it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.42it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.42it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.42it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.42it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.42it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.42it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.25it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.25it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.25it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.25it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.25it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.25it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  60%|██████    | 35/58 [00:01<00:00, 42.20it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:01<00:00, 42.20it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:01<00:00, 42.20it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:01<00:00, 42.20it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  60%|██████    | 35/58 [00:01<00:00, 42.20it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  60%|██████    | 35/58 [00:01<00:00, 42.20it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.19it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.73it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.73it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.73it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.73it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.73it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.73it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.13it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.20it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.20it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.20it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.20it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 36.15it/s]


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
    Generated text:  Andrew. I've been a piano player since I was just a child. My father had a piano that he could play on regular basis. I have a great passion for music. I love listening to music as it is a form of expression. It can be entertaining and relaxing. I have played the piano for 20 years now, and I have never been a bad player. I have played it for my whole life. The first time I tried to learn it, I was surprised to find that I was not good at it. I was not successful because I was not interested in playing it. But now I think I am good
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a neighboring country. During their meeting, the president makes a speech. A man from the neighboring country listens to the speech, and after 5 minutes, the man has heard 20% of the speech. After an hour and a half, the man has heard 65% of the speech. How many minutes did the speech last? Let's denote the total length of the speech as \( T \) minutes. According to the information given, after 5 minutes, the man has heard 20% of the speech. This means that in 5 minutes, he has heard \( 0.2T \
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. The capital of France is Paris. 
    
    A) England 
    B) Canada 
    C) Germany 
    D) Italy
    To determine the capital of France, let's first identify the correct answer from the given choices. The capital of France is Paris, which is the correct answer.
    
    Here's the step-by-step reasoning:
    
    1. Identify the capital of France. The capital of France is Paris.
    2. Compare the given choices with the capital of France. The choices are England, Canada, Germany, and Italy.
    3. Identify which of these choices corresponds to Paris.
    
    Therefore, the correct choice is:
    
    \boxed{B}
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but the experts agree that AI will continue to play an important role in the future of work and the economy. The AI market will grow at an impressive rate, and new research is emerging all the time. As the technology advances and is integrated into various sectors of our lives, we can expect to see an increase in the number of jobs that will be replaced by AI. In the meantime, AI will continue to play a vital role in the development of new and innovative solutions for various industries.
    
    AI is already being used in a wide variety of applications, such as image recognition, speech recognition, and natural language processing. These technologies are


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art, and cuisine. Paris is a popular tourist destination and a cultural hub for Europe. It is home to many world-renowned museums, theaters, and landmarks. The city is also known for its annual festivals and events, such as the Eiffel Tower Festival and the Louvre Festival. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations more effectively.
    
    2. Greater emphasis on ethical considerations: As AI becomes more prevalent in various industries, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI becomes more advanced, it is likely to be used in even more areas, including
    


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
    Generated text:  [Name] and I am a [job title] who has been dedicated to [career goal or passion area] for [career span] years. I am passionate about [mention a hobby or skill] and I strive to make a positive impact in the world by [mention a specific action or goal]. I am always [mention a positive trait or character trait]. I am always open to learning and have a [mention a skill or interest in sports, cooking, etc.]. I am excited to meet you and discuss my career goals and what I could bring to your team. What is your background and what is your most interesting or exciting
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city and one of the most populous cities in the world. It is also the oldest capital of the Kingdom of France. Paris is renowned for its historical landmarks, such as the Eiffel Tower and Notre-Dame Cathedral, and its role as a cultural and economic hub. Its status as a global leader in fashion and luxury goods further contributes to its appeal as a destination for tourists. Despite its size and influence, Paris has a unique and rich history, with its rich history dating back to the 12th century. It is also home to the world's oldest cathedral, Notre-Dame Cathedral, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see significant advancements in areas such as:
    
    1. Augmented and virtual reality: AI will continue to be used for virtual and augmented reality, enabling users to interact with the real world in immersive ways.
    
    2. Biometrics: The use of biometric data, such as facial recognition, fingerprint scans, and voice recognition, will become more prevalent as AI technologies continue to advance.
    
    3. Autonomous vehicles: AI will continue to play a critical role in autonomous vehicles, improving safety, reducing traffic congestion, and enhancing the efficiency of transportation.
    
    4. Robotics and automation: AI will continue to play a critical role in robotics and automation, enabling


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

    ].

     I

     am

     an

     [

    occupation

    ]

     who

     is

     passionate

     about

     [

    personal

     interest

     or

     hobby

    ]

     and

     always

     strive

     to

     [

    positive

     trait

     or

     motto

    ].

     I

     am

     confident

     in

     myself

     and

     believe

     in

     [

    value

     or

     belief

    ].

     I

     am

     here

     to

     share

     my

     experiences

     and

     help

     those

     who

     are

     looking

     for

     guidance

     or

     inspiration

    .


    I

    'm

     [

    name

    ]

     and

     I

    'm

     excited

     to

     meet

     you

    .

     Let

    's

     talk

     about

     [

    subject

    ].

     Looking

     forward

     to

     our

     conversation

    .

     [

    Name

    ]

     [

    Occup

    ation

    ]


    [

    Name

    ]

     -

     [

    Field

    ]


    [

    Occup

    ation

    ]

     -

     [

    Field

    ]

     [

    Name

    ]

     -

     [

    Field

    ]

     -

     [

    Subject

    ]


    Hi

    !

     How

    's

     it

     going

    ?

     I

     hope

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     rich

     cultural

     history

    .

     Paris

     is

     also

     renowned

     for

     its

     fashion

     industry

    ,

     which

     has

     made

     it

     a

     major

     tourist

     destination

    .

     France

    's

     capital

     city

     has

     a

     diverse

     population

     of

     over

     

    7

     million

     people

    ,

     making

     it

     a

     significant

     economic

     and

     cultural

     hub

     in

     Europe

    .

     It

     is

     a

     popular

     tourist

     destination

     with

     over

     

    3

    5

     million

     visitors

     annually

    .

     Paris

     is

     also

     known

     for

     its

     romantic

     atmosphere

    ,

     with

     its

     sunny

     days

    ,

     romantic

     evening

     scenes

    ,

     and

     picturesque

     Old

     Quarter

    .

     The

     city

     is

     famous

     for

     its

     art

    ,

     music

    ,

     and

     cuisine

    ,

     making

     it

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     Paris

    's

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     constantly

     evolving

    ,

     with

     potential

     future

     trends

     including

     increased

     focus

     on

     ethical

     considerations

    ,

     greater

     integration

     with

     human

     AI

    ,

     and

     the

     development

     of

     advanced

     machine

     learning

     algorithms

     capable

     of

     solving

     complex

     problems

     that

     are

     currently

     beyond

     human

     capabilities

    .

     In

     addition

    ,

     the

     continued

     development

     of

     artificial

     intelligence

     is

     likely

     to

     see

     continued

     progress

     in

     areas

     such

     as

     natural

     language

     processing

    ,

     computer

     vision

    ,

     and

     robotics

    ,

     as

     well

     as

     the

     development

     of

     new

     AI

     models

     that

     can

     address

     a

     variety

     of

     future

     challenges

    ,

     such

     as

     climate

     change

    ,

     pand

    emics

    ,

     and

     pand

    emics

    .

     Finally

    ,

     as

     AI

     becomes

     more

     integrated

     into

     everyday

     life

    ,

     there

     may

     be

     increased

     emphasis

     on

     the

     development

     of

     user

    -friendly

     interfaces

     and

     the

     creation

    



```python
llm.shutdown()
```
