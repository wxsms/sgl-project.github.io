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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.41it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.40it/s]


    2026-05-04 16:39:21,835 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 16:39:21] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:44,  4.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:44,  4.99s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:44,  4.99s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:44,  4.99s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:44,  4.99s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.97it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.97it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.97it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.97it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.97it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.97it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.97it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.97it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.97it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  7.97it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.49it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.20it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.20it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.20it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.20it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.20it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.20it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.20it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.20it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.20it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.20it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 20.20it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.18it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.18it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.18it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 33.50it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 33.50it/s] Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  31%|███       | 18/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.94it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.94it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.94it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.94it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.94it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.94it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.42it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.42it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.42it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.42it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.42it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.42it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.78it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.51it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.80it/s]

    Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=16 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.02it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 40.17it/s]


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
    Generated text:  Darius and I am a very dedicated mom with two wonderful children. I have a great job working with children and parents to help them feel secure and happy. I have 6 children: 5 boys and 1 girl. They are all very different in temperament and personality. Some are very energetic and outgoing, while others are very reserved and shy.
    I love being in nature and trying different foods, especially the ones from Vietnam.
    I love cooking, learning new foods, trying new recipes and snacks, and enjoying family dinners together.
    I'm always looking for new places to visit and new experiences, and I love doing so with my kids
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing to send a team of scientists to Antarctica to search for ice and other scientific data. The team will be on a team of ten people. If the president has selected 6 members of the team, how many members are left for the other team members to join?
    The president has already selected 6 members of the team, making it 10 members total. If 6 members are selected, then 10 - 6 = 4 members are left for the other team members to join. So, there are 4 members left to join the team.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located in the northwestern part of the country and is the capital of France. It is the largest and most populous city in the European Union and the world’s 29th most populous city.
    It is located in the Île de France (the island of France) and is situated on the Left Bank of the Seine. The Seine and the river landscape surrounding Paris constitute the city’s most notable natural features, as do its cultural and historical contributions.
    The history of Paris dates back to the 6th century, to the time when it was the seat of the Roman Empire. It was an important city
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it also poses many challenges. In this post, we take a look at the challenges and the solutions that the AI industry will face in the future.
    
      1. Ethical issues: One of the biggest challenges that the AI industry faces is the ethical issues surrounding the development and use of AI. In some cases, AI can lead to unintended consequences that could have serious impacts on people and society. For example, autonomous vehicles that are programmed to make decisions that could lead to accidents may not fully understand the consequences of their actions and could cause harm to others.
      2. Bias and Fairness: Another major challenge that


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling metropolis with a diverse population and is a major center for art, culture, and commerce. It is also home to many famous landmarks and attractions, including the Palace of Versailles and the Arc de Triomphe. Paris is a city that has been a major center of European culture and politics for centuries. Its status as the capital of France has made it a major hub for international
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    2. Greater integration with other technologies: AI is already being integrated into a wide range of technologies, including smart homes, self-driving cars, and virtual assistants. As these technologies continue to evolve, we may see even more integration between AI and other technologies.
    
    3. Increased use of AI in
    


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
    Generated text:  [Name], and I'm a [Type] with [Experience Level] years of experience in [Industry or field]. I'm passionate about [My Personal Interest], and I believe that [Motivation] will help me achieve my goals. I am always eager to learn and adapt to new challenges, and I'm always willing to help others improve their skills. My unique skillset includes [Your Unique Skillset], and I enjoy spending time outdoors, practicing my guitar, and exploring new places. I am a [Positive Traits] person, and I believe in [My Core Values] as the guiding principles of my character. Thank you for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic city known for its vibrant culture, fine art, and rich history. It is the third-largest city in Europe and is the political, cultural, and economic heart of France. Paris is home to many world-renowned landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, making it a popular destination for tourists and locals alike. Its status as a major European capital has made it a major influence on the French culture and politics. The city is also famous for its cuisine, including famous dishes like croissants and foie gras. Paris is a unique and distinctive city that offers visitors
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  predicted to be in a state of constant development and evolution, driven by a variety of factors including advancements in data and computing power, greater reliance on machine learning and artificial intelligence, and increasing complexity and sophistication of AI systems. Some of the potential future trends in AI include:
    
    1. Increased integration of AI into diverse industries: With the increasing importance of AI in various industries, such as healthcare, finance, and transportation, we can expect to see greater integration and adoption of AI technologies in these sectors. AI will be integrated into various processes and systems to improve efficiency, reduce costs, and enhance productivity.
    
    2. Greater reliance on AI in self-driving


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

     [

    Age

    ].

     I

     am

     a

     [

    Job

     Title

    ]

     with

     [

    Company

     Name

    ]

     for

     [

    Role

    ].

     I

     have

     [

    Number

     of

     Years

     Experience

    ]

     years

     of

     experience

     in

     [

    Field

    ].

     I

     am

     currently

     [

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     I

     am

     passionate

     about

     [

    personal

     interest

     or

     hobby

    ].

     I

     am

     [

    Level

     of

     commitment

    ]

     to

     my

     work

    .

     I

     have

     [

    Number

     of

     Projects

    ]

     projects

     completed

    ,

     with

     [

    Number

     of

     Accom

    pl

    ishments

    ]

     successes

    .

     I

     am

     [

    Level

     of

     loyalty

     to

     the

     company

    ].

     I

     am

     available

     to

     discuss

     any

     questions

     or

     concerns

     you

     may

     have

    .
    


    Please

     summarize

     your

     personal

     experience

     and

     achievements

     related

     to

     [

    Field

    /

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     oldest

     capital

     city

     in

     Europe

    ,

     with

     an

     impressive

     history

     dating

     back

     to

     ancient

     times

    .

     Paris

     is

     famous

     for

     its

     rich

     culture

    ,

     beautiful

     architecture

    ,

     and

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     home

     to

     many

     world

    -ren

    owned

     museums

    ,

     galleries

    ,

     and

     theaters

    ,

     and

     is

     also

     known

     for

     its

     lively

     food

     scene

     and

     wine

     culture

    .

     It

     is

     a

     city

     that

     continues

     to

     thrive

     as

     a

     center

     of

     global

     politics

    ,

     culture

    ,

     and

     entertainment

    .

     Paris

     is

     a

     truly

     unique

     and

     fascinating

     city

     that

     is

     a

     must

    -

    visit

     for

     anyone

     interested

     in

     French

     culture

     and

     history

    .

     According

     to

     the

     passage

    ,

     what

     is

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     currently

     dominated

     by

     the

     development

     of

     deep

     learning

    ,

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     robotics

    .

     AI

     is

     expected

     to

     continue

     to

     evolve

     and

     improve

     in

     many

     ways

     in

     the

     coming

     years

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

     focus

     on

     ethical

     AI

    :

     There

     is

     growing

     concern

     about

     the

     ethical

     implications

     of

     AI

     and

     the

     potential

     for

     unintended

     consequences

    .

     As

     AI

     becomes

     more

     advanced

    ,

     there

     will

     be

     an

     increased

     focus

     on

     developing

     AI

     that

     is

     more

     transparent

    ,

     accountable

    ,

     and

     accountable

    .
    


    2

    .

     Continued

     improvements

     in

     AI

     performance

    :

     AI

     is

     expected

     to

     continue

     to

     improve

     its

     performance

     and

     capabilities

     in

     many

     ways

    .

     This

     includes

     improvements

     in

     accuracy

    ,

     speed

    ,

     and

     efficiency

    



```python
llm.shutdown()
```
