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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<05:43,  6.03s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<05:43,  6.03s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:06<05:43,  6.03s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:06<05:43,  6.03s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:06<05:43,  6.03s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:48,  1.08it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:48,  1.08it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:06<00:48,  1.08it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:06<00:48,  1.08it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:06<00:48,  1.08it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:06<00:48,  1.08it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:06<00:48,  1.08it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:15,  2.96it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:15,  2.96it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:15,  2.96it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:06<00:15,  2.96it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:06<00:15,  2.96it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:06<00:15,  2.96it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:06<00:15,  2.96it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:06<00:15,  2.96it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:06<00:15,  2.96it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:06<00:15,  2.96it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:05,  6.73it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:05,  6.73it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:05,  6.73it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:06<00:05,  6.73it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:06<00:05,  6.73it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:06<00:05,  6.73it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:06<00:05,  6.73it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:06<00:05,  6.73it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:06<00:05,  6.73it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:06<00:02, 10.93it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:06<00:02, 10.93it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:06<00:02, 10.93it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:06<00:02, 10.93it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:06<00:02, 10.93it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:06<00:02, 10.93it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:06<00:02, 10.93it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:06<00:02, 10.93it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:06<00:02, 10.93it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:06<00:01, 16.10it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:06<00:01, 16.10it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:06<00:01, 16.10it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:06<00:01, 16.10it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:06<00:01, 16.10it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:06<00:01, 16.10it/s]

    Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:06<00:01, 16.10it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:06<00:01, 16.10it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:06<00:01, 16.10it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:06<00:01, 16.10it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 23.07it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 23.07it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 23.07it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 23.07it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 23.07it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:06<00:00, 23.07it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:06<00:00, 23.07it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:06<00:00, 23.07it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:06<00:00, 23.07it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:06<00:00, 23.07it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 31.02it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 31.02it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 31.02it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:06<00:00, 31.02it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:06<00:00, 31.02it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.96 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.93 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.93 GB):   3%|▎         | 2/58 [00:00<00:03, 17.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.65 GB):   3%|▎         | 2/58 [00:00<00:03, 17.25it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.46 GB):   3%|▎         | 2/58 [00:00<00:03, 17.25it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.46 GB):   7%|▋         | 4/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.95 GB):   7%|▋         | 4/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.94 GB):   7%|▋         | 4/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.93 GB):   7%|▋         | 4/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.93 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.93 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.93 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.92 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.42it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=70.92 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.92 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.91 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.91 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.91 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.91 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.47 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.90it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=70.47 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.47 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.47 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.45 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.44it/s]Capturing num tokens (num_tokens=960 avail_mem=70.46 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.44it/s] Capturing num tokens (num_tokens=896 avail_mem=70.46 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.44it/s]Capturing num tokens (num_tokens=832 avail_mem=70.45 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.44it/s]Capturing num tokens (num_tokens=832 avail_mem=70.45 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.15it/s]Capturing num tokens (num_tokens=768 avail_mem=70.45 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.15it/s]Capturing num tokens (num_tokens=704 avail_mem=70.45 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.15it/s]Capturing num tokens (num_tokens=640 avail_mem=70.44 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.15it/s]

    Capturing num tokens (num_tokens=576 avail_mem=70.44 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.15it/s]Capturing num tokens (num_tokens=512 avail_mem=70.43 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.15it/s]Capturing num tokens (num_tokens=512 avail_mem=70.43 GB):  50%|█████     | 29/58 [00:00<00:00, 38.40it/s]Capturing num tokens (num_tokens=480 avail_mem=70.44 GB):  50%|█████     | 29/58 [00:00<00:00, 38.40it/s]Capturing num tokens (num_tokens=448 avail_mem=70.44 GB):  50%|█████     | 29/58 [00:00<00:00, 38.40it/s]Capturing num tokens (num_tokens=416 avail_mem=70.44 GB):  50%|█████     | 29/58 [00:00<00:00, 38.40it/s]Capturing num tokens (num_tokens=384 avail_mem=70.44 GB):  50%|█████     | 29/58 [00:00<00:00, 38.40it/s]Capturing num tokens (num_tokens=352 avail_mem=70.43 GB):  50%|█████     | 29/58 [00:01<00:00, 38.40it/s]Capturing num tokens (num_tokens=352 avail_mem=70.43 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]Capturing num tokens (num_tokens=320 avail_mem=70.42 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]Capturing num tokens (num_tokens=288 avail_mem=70.42 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]Capturing num tokens (num_tokens=256 avail_mem=70.42 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]

    Capturing num tokens (num_tokens=240 avail_mem=70.42 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]Capturing num tokens (num_tokens=224 avail_mem=70.41 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]Capturing num tokens (num_tokens=224 avail_mem=70.41 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=208 avail_mem=70.41 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=192 avail_mem=70.41 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=176 avail_mem=70.40 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=160 avail_mem=70.40 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=144 avail_mem=70.40 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.78it/s]Capturing num tokens (num_tokens=144 avail_mem=70.40 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.44it/s]Capturing num tokens (num_tokens=128 avail_mem=70.40 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.44it/s]Capturing num tokens (num_tokens=112 avail_mem=70.40 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.44it/s]Capturing num tokens (num_tokens=96 avail_mem=70.39 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.44it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=70.39 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.44it/s]Capturing num tokens (num_tokens=64 avail_mem=70.38 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.44it/s]Capturing num tokens (num_tokens=64 avail_mem=70.38 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.82it/s]Capturing num tokens (num_tokens=48 avail_mem=70.38 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.82it/s]Capturing num tokens (num_tokens=32 avail_mem=70.38 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.82it/s]Capturing num tokens (num_tokens=28 avail_mem=70.37 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.82it/s]Capturing num tokens (num_tokens=24 avail_mem=70.37 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.82it/s]Capturing num tokens (num_tokens=20 avail_mem=70.37 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.82it/s]Capturing num tokens (num_tokens=20 avail_mem=70.37 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.44it/s]Capturing num tokens (num_tokens=16 avail_mem=70.37 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.44it/s]Capturing num tokens (num_tokens=12 avail_mem=70.36 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.44it/s]Capturing num tokens (num_tokens=8 avail_mem=70.36 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.44it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=70.35 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.44it/s]Capturing num tokens (num_tokens=4 avail_mem=70.35 GB): 100%|██████████| 58/58 [00:01<00:00, 37.68it/s]


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
    Generated text:  Liza. I'm from Wales, and I have been living in London for about 5 years. I'm now an English teacher. My daughter is a student at a primary school in London. I have a problem with my walking. I keep falling down. I am not used to walking so fast. I don't know how to fix it. I have tried going to the hospital, but I still don't know what the problem is. What should I do?
    A. Go to the sports center to play some games and then go to the hospital.
    B. Go to the hospital and try the best medicine.
    C. Ask
    ===============================
    Prompt: The president of the United States is
    Generated text:  a(n) ________. A. president of the United States is a(n) ________. B. president of the United States is a(n) ________. C. president of the United States is a(n) ________. A president of the United States is a(n) **Sovereign**. **Sovereign** is a concept that refers to a person who has the authority to make and enforce laws for the state and country. The president of the United States is the head of government and the head of the executive branch of the U.S. government. They are responsible for leading the country and making decisions that affect the entire population
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It has a population of about 2.1 million people. Paris is also known as the City of Light, the City of Lights, and the City of Light. The city’s name is derived from the French words for “lights” (lumières).
    Paris is located in the south of France, on the Mediterranean Sea. The city is very old, having been inhabited since ancient times. It is one of the most important cities in France, and it is one of the most visited cities in the world.
    Paris has a long history, dating back to the time of the Roman Empire. It was the capital of the
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the developers. To help you understand the most important features of AI, we have summarised the key features of AI, which are the main elements that define AI and provide the basis for all the other features. These include: Artificial intelligence, Machine learning, Deep learning, Natural language processing, Computer vision, Computer graphics, Robotics, Speech recognition, Perception, Speech synthesis, and so on.
    
    Artificial intelligence (AI) is the name given to the ability of a machine to perform tasks that would normally require human intelligence, such as learning, reasoning, planning and decision-making. AI can be applied to a wide range of


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? As an AI language model, I don't have a physical presence, but I'm always ready to assist you with any questions or tasks you may have. How can I help you today? Let's get started! [Name] [Company Name] is a [brief description of the company]. We specialize in [specific services or products]. I'm excited to learn more about your career and how I can help you achieve your goals. [Name] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is the largest city in France and the third-largest city in the world by population. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the Middle Ages. The city is known for its vibrant nightlife, art scene, and world-class museums and attractions. It is also home to many famous landmarks such as Notre-Dame Cathedral and the Palace of Versailles. Paris is a major transportation hub and a major tourist destination, attracting millions of visitors each year. It is a city of contrasts, with its modern architecture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do. This could lead to more sophisticated forms of AI, such as those that can learn and adapt to new situations on their own.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in even more areas, including diagnosis, treatment, and patient care.
    
    3. Greater
    


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
    Generated text:  [Name] and I'm a professional marketing manager. I have a passion for helping businesses grow and I'm dedicated to delivering results. I'm always looking for new opportunities to learn and grow as a professional. What do you do?
    
    [Name]: Thanks for asking. I'm a marketing manager with a proven track record of success. I've worked for [Company] for several years and I've helped businesses like [Company] to achieve their goals. I'm always looking for new ways to help businesses grow and I'm passionate about delivering results. 
    
    I enjoy working with clients and helping them understand the unique needs of their business. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, and it is known for its romantic architecture, diverse cuisine, and vibrant culture. It is a historic city with a rich history dating back to ancient times, and it has a population of approximately 2.1 million people. The city is home to the Eiffel Tower, a landmark that has stood for over 100 years, and it is also home to the Louvre Museum, which houses one of the world's largest collections of art and artifacts. Paris is a city of contrasts, with its diverse neighborhoods and vibrant nightlife, and it is a popular tourist destination for visitors from around the world. The city has
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and will depend on a number of factors, including technological progress, economic and social change, and human attitudes towards AI. Here are some potential trends that could shape AI in the coming decades:
    
    1. AI will become more pervasive: As AI continues to advance, we will see more and more of our daily activities performed by machines that can learn and adapt to new situations. This could lead to a more pervasive and widespread use of AI, with machines playing a larger role in our lives.
    
    2. AI will become more capable: As AI continues to evolve, we may see more complex and sophisticated machines that can perform a wide range of tasks


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

    ],

     and

     I

     am

     a

     [

    Occup

    ation

    ]

    !
    


    I

    'm

     a

     [

    Occup

    ation

    ]

     who

     specializes

     in

     [

    Skill

     or

     Expert

    ise

    ].

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     develop

    .

     What

     exc

    ites

     me

     most

     about

     my

     work

     is

     [

    What

     exc

    ites

     you

     most

     about

     your

     job

     or

     occupation

    ].
    


    If

     you

    're

     interested

     in

     learning

     more

     about

     me

     or

     trying

     out

     my

     work

    ,

     please

     do

     not

     hesitate

     to

     reach

     out

    .

     I

    'd

     love

     to

     hear

     from

     you

    !
    


    Let

    's

     make

     a

     connection

    !

     

    📞

    ✨

    
    


    ---
    


    *

    Note

    :

     Replace

     [

    Your

     Name

    ],

     [

    Occup

    ation

    ],

     [

    Skill

     or

     Expert

    ise

    ],

     and

     [

    What

    
    
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

     with

     an

     estimated

     population

     of

     around

     

    2

    .

    3

     million

     people

     in

     the

     

    2

    0

    1

    9

     population

     census

    .

     The

     city

     is

     home

     to

     the

     French

     capital

    ,

     Paris

    ,

     which

     is

     also

     known

     as

     the

     Lou

    vre

    ,

     and

     is

     the

     seat

     of

     government

    ,

     culture

    ,

     arts

    ,

     and

     commerce

     for

     France

    .

     The

     French

     capital

     city

     is

     located

     in

     the

     center

     of

     the

     country

    ,

     near

     the

     Atlantic

     Ocean

    .

     It

     is

     also

     home

     to

     many

     famous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

     Palace

     of

     Vers

    ailles

    ,

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     It

     is

     a

     major

     cultural

     and

     economic

     hub

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     highly

     dependent

     on

     the

     level

     of

     integration

     of

     AI

     into

     our

     daily

     lives

    .

     Here

     are

     some

     possible

     trends

     we

     can

     expect

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     automation

    :

     AI

     has

     the

     potential

     to

     completely

     automate

     many

     tasks

    ,

     including

     those

     that

     require

     human

     judgment

     or

     expertise

    .

     This

     could

     lead

     to

     job

     displacement

     and

     create

     new

     opportunities

     for

     new

     industries

     and

     professions

    .
    


    2

    .

     Improved

     health

     and

     wellness

    :

     AI

     could

     be

     used

     to

     monitor

     and

     improve

     the

     health

     of

     individuals

    ,

     enabling

     them

     to

     make

     better

     health

     decisions

     and

     improve

     their

     overall

     well

    -being

    .
    


    3

    .

     Enhanced

     personalized

     healthcare

    :

     AI

     could

     be

     used

     to

     analyze

     large

     amounts

     of

     medical

     data

     and

     provide

     more

     accurate

     diagnoses

     and

     treatment

    



```python
llm.shutdown()
```
