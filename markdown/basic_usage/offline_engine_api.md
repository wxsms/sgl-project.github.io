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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.21it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.20it/s]


    2026-05-15 08:42:16,905 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 08:42:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.33it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.70it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.63it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.69it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.69it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.69it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.69it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.69it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.69it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.69it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.69it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.69it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.69it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.69it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.67it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.67it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.67it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.67it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.67it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.67it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.67it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.69it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.69it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   9%|▊         | 5/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.12 GB):   9%|▊         | 5/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.09 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.99it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=75.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.08 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.15 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.15 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.15 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.15 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.46it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.44it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.16it/s]Capturing num tokens (num_tokens=960 avail_mem=73.97 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.16it/s] Capturing num tokens (num_tokens=896 avail_mem=73.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.16it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.16it/s]Capturing num tokens (num_tokens=768 avail_mem=73.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.16it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.16it/s]

    Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=416 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=416 avail_mem=73.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.81it/s]Capturing num tokens (num_tokens=384 avail_mem=73.94 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.81it/s]Capturing num tokens (num_tokens=352 avail_mem=73.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.81it/s]Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.81it/s]Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.81it/s]

    Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.81it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=208 avail_mem=73.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=192 avail_mem=73.92 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=144 avail_mem=73.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=128 avail_mem=73.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.22it/s]

    Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.22it/s] Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  81%|████████  | 47/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=80 avail_mem=73.90 GB):  81%|████████  | 47/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  81%|████████  | 47/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  81%|████████  | 47/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=32 avail_mem=73.89 GB):  81%|████████  | 47/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  81%|████████  | 47/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=16 avail_mem=73.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.60it/s]

    Capturing num tokens (num_tokens=8 avail_mem=73.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.60it/s] Capturing num tokens (num_tokens=8 avail_mem=73.87 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 38.04it/s]


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
    Generated text:  Joseph, and I'm a computer science student at Dartmouth College. I'm majoring in Computer Science. I'm learning how to program and I'm learning to create things. I want to be a big boss of my own, but I'm not sure how to make it work. Can you tell me what I should do?
    
    Of course, I'd be happy to help! To start, you should define your goals for yourself. What do you want to achieve with your programming skills? Are you looking to build a website, create a game, or something else entirely? Having a clear goal will help you determine what path to take
    ===============================
    Prompt: The president of the United States is
    Generated text:  a candidate for a non-executive director position on the board of directors of a publicly traded company. The company's common stock has a market value of $25.00 per share and currently trades at $22.00. The president's share value is $32.00, and he proposes to sell 5% of the shares to his candidate for the position. Determine the net return on investment for the candidate if he sells the shares in the market, assuming no other fees or expenses.
    To determine the net return on investment for the candidate, we need to follow these steps:
    
    1. **Calculate the
    ===============================
    Prompt: The capital of France is
    Generated text:  _________. A. Paris B. London C. Moscow
    A. Paris
    
    The capital of France is Paris. This is a correct answer based on historical facts and official data. If you have any other questions about French cities or capitals, feel free to ask!
    ===============================
    Prompt: The future of AI is
    Generated text:  coming, and it’s happening in the most unexpected ways. Here’s why:
    
    The world is experiencing an unprecedented shift in how we interact with technology. Mobile and cloud computing are pervasive and our digital devices, connected devices, and internet are providing us with an unprecedented amount of information. As a result, AI is becoming more important than ever before, and it will play an increasingly important role in our lives.
    
    Here are 6 ways AI is changing the way we interact with the world.
    
      1. Personalized Assistance
    
    AI is being used to personalize and assist with a wide range of tasks and activities, from scheduling appointments to providing advice


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


    Generated text:  [Name] and I am a [occupation] who has been working in the [industry] for [number] years. I am passionate about [reason for passion], and I am always looking for ways to [action or achievement]. I am a [character trait or quality] who is always [description of a trait or quality]. I am [character description], and I am [character trait or quality]. I am [character description], and I am [character trait or quality]. I am [character description], and I am [character trait or quality]. I am [character description], and I am [character trait or quality]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a popular tourist destination and a major economic center. The city is home to many world-renowned museums, including the Louvre and the Musée d'Orsay, and is known for its rich history and cultural heritage. Paris is a vibrant and diverse city with a rich cultural scene, and is a popular destination for tourists and locals alike. The city is also home to many important political and economic institutions, including the French Parliament and the French National Assembly. Overall, Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI will continue to automate many tasks, from manufacturing to customer service, and will likely become more efficient and accurate.
    
    2. Enhanced human-AI collaboration: As AI becomes more capable, it will be able to work more closely with humans, improving efficiency and productivity.
    
    3. AI ethics and privacy concerns: As AI becomes more advanced, there will be increasing concerns about its ethical implications and potential privacy violations.
    
    4. AI for social good: AI will be used to address social and environmental issues, such as climate change, healthcare, and education.
    
    5. AI for personal
    


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
    Generated text:  [insert name] and I specialize in [insert specialty or expertise]. I have a passion for [insert passion or interest]. I’m always eager to learn and I’m always looking for opportunities to grow and develop as a professional. I thrive on challenges and never give up on my goal of achieving success. I believe in the power of perseverance and never give up when I face any obstacle. I’m excited to meet you and learn more about you. Can you provide me with some more information about your specialization or expertise? Hello! Thank you for asking! My specialty is in [insert specialty or expertise], which is a unique and innovative field
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and home to the iconic Eiffel Tower. Paris is known for its rich history, art, and cuisine, and is often referred to as the "city of love" for its romantic atmosphere. The city is also home to some of the world's most famous landmarks, including the Louvre Museum and Notre Dame Cathedral. Paris is a bustling hub of activity, with its famous museums, theaters, and restaurants attracting millions of visitors each year. The city's location in the Paris Basin provides access to the Mediterranean Sea, making it a popular destination for tourists and travelers alike. Its climate, characterized
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of different trends and developments that will shape its development and impact on society. Here are some possible future trends in AI:
    
    1. Increased availability of AI-powered tools and technologies: As AI technologies become more sophisticated and capable, we may see an increased availability of AI-powered tools and technologies that can help us solve complex problems and automate many of our everyday tasks. This could include tools for data analysis, automation of repetitive tasks, and even virtual assistants that can help us navigate the digital world.
    
    2. Personalized AI: AI will also become more capable of understanding and adapting to our personal preferences and behaviors, leading to more


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

     here

    ],

     and

     I

     am

     a

     [

    insert

     profession

     here

    ].

     I

     believe

     in

     [

    insert

     something

     positive

    ,

     like

     being

     kind

     or

     kind

    red

     spirit

    ].

     I

     am

     always

     ready

     to

     help

     others

     and

     have

     a

     great

     time

    .

     I

     also

     have

     a

     good

     sense

     of

     humor

     and

     enjoy

     making

     others

     laugh

    .

     I

     strive

     to

     be

     a

     positive

     example

     for

     others

     and

     contribute

     to

     a

     better

     world

    .

     I

     am

     always

     learning

     and

     growing

    ,

     and

     I

     am

     always

     open

     to

     new

     experiences

    .

     So

     if

     you

    're

     ready

     to

     make

     friends

    ,

     get

     advice

    ,

     or

     just

     want

     to

     have

     some

     fun

    ,

     I

    'm

     your

     go

    -to

     person

    !

     

    💪

     #

    char

    ism

    aware

     #

    self

    ie

    challenge

     #

    g

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    la

     pop

    ."

     Its

     name

     comes

     from

     the

     Latin

     "

    pop

    ulus

    ,"

     meaning

     "

    the

     people

    ."
    


    That

    's

     correct

    !

     Paris

    ,

     officially

     known

     as

     "

    la

     pop

    ,"

     is

     the

     capital

     city

     of

     France

     and

     the

     largest

     city

     in

     Europe

    .

     It

    's

     the

     

    1

    2

    th

     largest

     city

     in

     the

     world

     by

     population

    ,

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

    .

     The

     city

     is

     known

     for

     its

     elegant

     architecture

    ,

     iconic

     landmarks

     like

     the

     E

    iff

    el

     Tower

    ,

     and

     a

     vibrant

     cultural

     scene

    .

     It

    's

     also

     home

     to

     several

     famous

     museums

    ,

     such

     as

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     hosts

     numerous

     world

    -class

     events

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

     and

     exciting

    ,

     with

     the

     potential

     to

     change

     the

     way

     we

     live

     and

     work

    .

     Some

     possible

     trends

     that

     we

     can

     expect

     to

     see

     in

     the

     near

     and

     long

     term

     include

    :
    


    1

    .

     Improved

     privacy

     and

     security

    :

     As

     AI

     continues

     to

     become

     more

     advanced

     and

     pervasive

    ,

     we

     will

     need

     to

     address

     privacy

     concerns

     and

     cybersecurity

     threats

    .

     This

     may

     require

     advancements

     in

     technologies

     like

     blockchain

    ,

     which

     can

     help

     secure

     data

     and

     protect

     against

     cyber

     attacks

    .
    


    2

    .

     Increased

     automation

     and

     efficiency

    :

     AI

     will

     continue

     to

     play

     an

     increasing

     role

     in

     our

     daily

     lives

    ,

     and

     automation

     is

     likely

     to

     increase

    .

     As

     AI

     becomes

     more

     capable

     and

     efficient

    ,

     it

     may

     take

     on

     more

     and

     more

     of

     the

     tasks

     that

    



```python
llm.shutdown()
```
