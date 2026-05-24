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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.87it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.85it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.85it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.85it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.85it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.85it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.85it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.85it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.85it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.85it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.85it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.70it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.70it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.70it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.70it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.70it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.70it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.70it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.70it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.70it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.70it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.70it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 38.60it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 38.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:03, 17.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:03, 17.29it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:03, 17.29it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:03, 17.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.50 GB):   9%|▊         | 5/58 [00:00<00:02, 21.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.49 GB):   9%|▊         | 5/58 [00:00<00:02, 21.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 21.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 21.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 21.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.48 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.47 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.47 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.98it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=73.47 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.46 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.46 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.45 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.45 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.45 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.44 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.44 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.95it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.44 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.42 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.95it/s]

    Capturing num tokens (num_tokens=960 avail_mem=73.44 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.95it/s] Capturing num tokens (num_tokens=896 avail_mem=73.43 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.95it/s]Capturing num tokens (num_tokens=832 avail_mem=73.43 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.95it/s]Capturing num tokens (num_tokens=832 avail_mem=73.43 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.59it/s]Capturing num tokens (num_tokens=768 avail_mem=73.43 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.59it/s]Capturing num tokens (num_tokens=704 avail_mem=73.42 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.59it/s]Capturing num tokens (num_tokens=640 avail_mem=73.42 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.59it/s]Capturing num tokens (num_tokens=576 avail_mem=73.42 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.59it/s]Capturing num tokens (num_tokens=512 avail_mem=73.40 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.59it/s]Capturing num tokens (num_tokens=512 avail_mem=73.40 GB):  50%|█████     | 29/58 [00:00<00:00, 41.14it/s]Capturing num tokens (num_tokens=480 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 41.14it/s]

    Capturing num tokens (num_tokens=448 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 41.14it/s]Capturing num tokens (num_tokens=416 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 41.14it/s]Capturing num tokens (num_tokens=384 avail_mem=73.41 GB):  50%|█████     | 29/58 [00:00<00:00, 41.14it/s]Capturing num tokens (num_tokens=352 avail_mem=73.41 GB):  50%|█████     | 29/58 [00:00<00:00, 41.14it/s]Capturing num tokens (num_tokens=352 avail_mem=73.41 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.58it/s]Capturing num tokens (num_tokens=320 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.58it/s]Capturing num tokens (num_tokens=288 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.58it/s]Capturing num tokens (num_tokens=256 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.58it/s]Capturing num tokens (num_tokens=240 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.58it/s]Capturing num tokens (num_tokens=224 avail_mem=73.39 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=224 avail_mem=73.39 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.69it/s]Capturing num tokens (num_tokens=208 avail_mem=73.39 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.69it/s]

    Capturing num tokens (num_tokens=192 avail_mem=73.39 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.69it/s]Capturing num tokens (num_tokens=176 avail_mem=73.38 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.69it/s]Capturing num tokens (num_tokens=160 avail_mem=73.38 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.69it/s]Capturing num tokens (num_tokens=144 avail_mem=73.38 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.69it/s]Capturing num tokens (num_tokens=144 avail_mem=73.38 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.54it/s]Capturing num tokens (num_tokens=128 avail_mem=73.38 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.54it/s]Capturing num tokens (num_tokens=112 avail_mem=73.37 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.54it/s]Capturing num tokens (num_tokens=96 avail_mem=73.37 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.54it/s] Capturing num tokens (num_tokens=80 avail_mem=73.37 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.54it/s]Capturing num tokens (num_tokens=64 avail_mem=73.36 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.54it/s]

    Capturing num tokens (num_tokens=64 avail_mem=73.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=48 avail_mem=73.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=32 avail_mem=73.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=28 avail_mem=73.35 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=24 avail_mem=73.35 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=20 avail_mem=73.34 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=20 avail_mem=73.34 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=16 avail_mem=73.34 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=12 avail_mem=73.34 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=8 avail_mem=73.34 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.02it/s] Capturing num tokens (num_tokens=4 avail_mem=73.33 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.02it/s]

    Capturing num tokens (num_tokens=4 avail_mem=73.33 GB): 100%|██████████| 58/58 [00:01<00:00, 39.54it/s]


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
    Generated text:  David, and I'm a 16-year-old college student with a passion for learning and knowledge. I am an avid reader, and I love exploring new concepts and concepts that I haven't explored before. I am also passionate about technology, and I enjoy learning about the latest gadgets and products in the tech industry. I enjoy sharing my knowledge with others and helping them learn as well. I have a keen interest in history and politics, and I am always eager to learn about new developments in these areas. I have always been passionate about being involved in the community, and I am always looking for ways to make a positive impact in my
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the Vice President. The vice president has a body of 60 people, and he is responsible for making 8 decisions. If each decision takes 2 minutes to make, how many minutes will it take for the Vice President to complete all the decisions?
    
    To determine how many minutes it will take for the Vice President to complete all the decisions, we can follow these steps:
    
    1. Identify the number of decisions the Vice President is responsible for making.
    2. Determine the time it takes to make one decision.
    3. Calculate the total time required to complete all the decisions by multiplying the number of decisions by the time per decision
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Brussels C. Lyon D. Lille
    Answer:
    A
    
    Which of the following situations can be classified as a risk event?
    A. An employee of the company is bitten by a rodent.
    B. A worker at the factory dies from a car accident.
    C. A person traveling alone in a car crashes into a car heading in the opposite direction.
    D. A company's computer system malfunctions and loses its data.
    Answer:
    D
    
    A. Incorrect
    B. Correct
    C. Uncertain
    D. None of the above options A to D are correct.
    
    Answer:
    B
    
    Given
    ===============================
    Prompt: The future of AI is
    Generated text:  digital.
    
    That's the overarching theme of the American Association for the Advancement of Science's recent "AI: Science for All" conference. The meeting was held on Thursday in the Kennedy Center for the Arts in Washington, D.C., and featured presentations on the latest in research from both the private and public sectors.
    
    Here are the key points of the conference:
    
      1. AI will be used in ways we don't think now
      2. Human traits and knowledge will be preserved and used for future generations
      3. Technology will be applied in fields that were previously impossible
    
    The AI field is not just a technical


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, art, and cuisine, and is a major center of politics, science, and culture in France. It is also home to many famous French artists and writers, including Pablo Picasso, Vincent van Gogh, and André Breton. The city is known for its diverse population,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is likely to play a greater role in healthcare, with machines being used to diagnose and treat diseases, as well as to assist in patient care and treatment planning.
    
    4. Greater use of
    


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
    Generated text:  [insert name], and I am a [insert age] year old [insert profession or occupation] by [insert place of birth]. I have always loved learning and always wanted to make a difference in the world. My passion for social justice and the fight for equality is fueled by my own experiences growing up in poverty, and I am determined to use my skills and knowledge to create positive change for myself and others. I'm looking forward to learning more about you and your journey to becoming an amazing person. Would you like to ask me about my life and work, or do you have anything specific you would like to know? 
    Your self
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the 19th- and 20th- century metropolis is known for its architectural diversity, including the Eiffel Tower and the Louvre Museum. The city is a cultural and economic hub, with a rich history dating back to the Roman Empire. The city is also home to numerous museums, including the Musée d'Orsay, the Musée Rodin, and the Musée d'Orsay. Its cuisine is also famous, with many world-renowned chefs cooking delicious dishes in the city's many restaurants. Paris is a city of contrasts, with a vibrant nightlife, a lively cultural scene,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting, with many possibilities and innovations shaping how we use and interact with this technology. Here are some of the key trends that are likely to shape the AI landscape in the coming years:
    
    1. Increased AI in Healthcare: AI has the potential to revolutionize the healthcare industry, with applications in diagnosis, treatment, and patient care. This could lead to better patient outcomes, faster diagnosis of diseases, and more personalized treatment plans.
    
    2. Autonomous Vehicles: As AI technology continues to improve, we could see more autonomous vehicles on the road, reducing the number of human drivers and increasing safety. However, this also raises concerns about job displacement and


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

     [

    insert

     occupation

    ].

     I

     love

     [

    insert

     a

     specific

     activity

    ,

     hobby

     or

     interest

    ]

     and

     I

     enjoy

     spending

     time

     with

     friends

     and

     family

    .

     I

     am

     always

     looking

     for

     new

     experiences

     and

     challenges

    ,

     and

     I

     am

     always

     eager

     to

     learn

     something

     new

    .

     I

     value

     teamwork

     and

     communication

    ,

     and

     I

     am

     always

     looking

     for

     ways

     to

     grow

     and

     develop

    .

     I

     am

     a

     person

     who

     is

     reliable

    ,

     honest

    ,

     and

     kind

    ,

     and

     I

     am

     always

     ready

     to

     lend

     a

     helping

     hand

     when

     needed

    .

     I

     am

     looking

     forward

     to

     meeting

     you

     and

     learning

     more

     about

     you

    !

     Let

    's

     have

     a

     conversation

     about

     our

     shared

     interests

    ,

     what

     we

     like

     to

     do

     together

    ,

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     capital

     of

     France

    .

     It

     is

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

     and

     is

     the

     seat

     of

     the

     French

     government

    ,

     parliament

    ,

     and

     the

     country

    's

     heart

     and

     most

     visited

     city

    .

     Paris

     is

     renowned

     for

     its

     historical

     landmarks

    ,

     museums

    ,

     and

     cultural

     institutions

    ,

     as

     well

     as

     its

     vibrant

     nightlife

     and

     fashion

     scene

    .

     The

     city

     is

     home

     to

     numerous

     world

    -ren

    owned

     museums

    ,

     such

     as

     the

     Lou

    vre

     and

     Mus

    ée

     d

    '

    Or

    say

    ,

     as

     well

     as

     restaurants

    ,

     shopping

     malls

    ,

     and

     a

     variety

     of

     cultural

     venues

    .

     Paris

     is

     a

     major

     global

     hub

     for

     finance

    ,

     media

    ,

     and

     entertainment

    ,

     attracting

     millions

     of

     visitors

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     highly

     dynamic

     and

     rapidly

     evolving

     field

    ,

     with

     a

     wide

     range

     of

     potential

     applications

     and

     innovations

     that

     could

     dramatically

     change

     our

     daily

     lives

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     speech

     recognition

    ,

     the

     possibilities

     for

     innovation

     and

     development

     are

     likely

     to

     continue

     to

     expand

    .
    


    2

    .

     Enhanced

     privacy

     and

     data

     security

    :

     As

     AI

     systems

     become

     more

     sophisticated

     and

     complex

    ,

     there

     will

     be

     increased

     concerns

     about

     privacy

     and

     data

     security

    .

     There

     may

     be

     emerging

     technologies

     and

     practices

     that

     address

     these

     issues

    ,

     such

     as

     AI

    -powered

     privacy

     protection

     tools

    



```python
llm.shutdown()
```
