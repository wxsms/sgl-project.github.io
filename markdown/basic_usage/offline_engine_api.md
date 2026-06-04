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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.48it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.06it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.06it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.06it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.06it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.06it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.06it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.06it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.06it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 20.82it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 20.82it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 20.82it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 20.82it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 20.82it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 20.82it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 20.82it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 20.82it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 20.82it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 20.82it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:04<00:00, 20.82it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 30.22it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 30.22it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 30.22it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.17it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.61it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.61it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.61it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 33.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 33.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 33.83it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 33.83it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 33.83it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.13it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.13it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.13it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.13it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.13it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.13it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.09it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.97it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.18it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 44.57it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.36it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.04it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.04it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 39.54it/s]


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
    Generated text:  Bill. I'm a college student. I've been living in New York City for the past two years. I'm studying the city with a classmate. He was born in New York City. He's been living there since his parents' death. We just moved in. I'm quite used to the city. I can't wait to start here. He's used to the city. He lives there all the time. He has all kinds of hobbies and interests. But he is completely different from me. He's now in New York City for two years. He's been living there all year. He has lots of friends.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a figurehead and does not have the power of the presidency. While the president does have the power to veto legislation, a presidential veto is never overridden. What is the power of the presidential veto?
    
    The power of the presidential veto is that it has the authority to override a bill passed by Congress. The president does not have the power to make or amend legislation, but can override a bill that Congress has passed by his own veto.
    
    Here's the step-by-step reasoning:
    
    1. **President of the United States**: The President of the United States is a figurehead and does not have the actual power to enforce legislation.
    2. **
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a very old city. It was founded in 789 A. D. and has a history of more than 1,000 years. It was the capital of the Empire of the Rhine from 962 A. D. to 1804 A. D. The city was destroyed by war many times, but it has never been completely destroyed and continues to grow. The city is very large. It has a population of about 6. 5 million and it is the second largest city in the world. It is known to be the world's largest city in terms
    ===============================
    Prompt: The future of AI is
    Generated text:  here
    
      • 10/21/2020
      • 2 minutes to read
    
    The world is evolving and the tech industry is also expanding. There are many topics that are about to see a lot of interest and discussion. Among them is AI, the technology that will change the world. Let’s talk about it.
    
    Who is AI?
    
    At its core, AI is a subset of computer science that involves creating software and algorithms that can perform certain tasks without human intervention. AI is becoming increasingly common, and many companies are trying to create their own artificial intelligence (AI) systems. This is happening because AI


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of French colonialism and the impact of the French Revolution. It is also home to many notable French artists, writers, and musicians. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has played a significant role
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased automation and robotics: AI is already being used in manufacturing, healthcare, and transportation, and we can expect to see even more automation and robotics in the future. This will lead to increased efficiency, reduced costs, and improved quality of life.
    
    2. Enhanced cognitive abilities: AI will continue to improve its ability to process and understand complex information, leading to new applications in fields such as education, healthcare, and finance.
    
    3. Personalization and customization: AI will enable more personalized and customized experiences for users, leading to increased efficiency and cost savings in various industries
    


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
    Generated text:  [Name], and I am [Age]. I have always loved [occupation], and I am dedicated to [what I do]. I am always [specific characteristic], and I am [specific characteristic], too. I am a [specific interest], and I am passionate about [related hobbies or activities]. I am [specific skill or ability], and I enjoy [related hobby or activity]. I am a [specific interest] and I love [related hobby or activity]. I am [specific interest] and I love [related hobby or activity]. I am [specific interest] and I love [related hobby or activity]. I am [specific interest
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history and diverse culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of potential innovations and opportunities for the industry to grow and thrive. Here are some possible trends in AI that could shape the future:
    
    1. Increased emphasis on ethics and compliance: With the rise of AI-powered systems that are used in industries such as finance, healthcare, and government, there is a growing need for systems that can handle ethical concerns and comply with regulations. This could lead to the development of more sophisticated ethical AI systems that can ensure that AI-powered systems are used in a responsible and fair manner.
    
    2. Increased use of AI for autonomous vehicles: As autonomous vehicles become more widely available, there is a potential for AI to play


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

     a

     [

    occupation

    ]

     who

     has

     a

     [

    job

     title

     or

     role

    ]

     in

     [

    company

     name

    ].

     I

     started

     my

     career

     in

     [

    year

    ],

     and

     I

     am

     currently

     [

    current

     role

    ].

     I

     enjoy

     [

    reason

     for

     being

     a

     self

    -employed

     freelancer

    ].

     I

     am

     passionate

     about

     [

    what

     I

     do

    ],

     and

     I

     look

     forward

     to

     creating

     unique

     solutions

     for

     my

     clients

    .

     What

     exc

    ites

     me

     the

     most

     is

     the

     opportunity

     to

     be

     my

     own

     boss

     and

     take

     on

     challenges

    ,

     while

     also

     learning

     new

     skills

     and

     making

     my

     own

     mark

     in

     the

     industry

    .

     Thank

     you

     for

     asking

    ,

     and

     I

     look

     forward

     to

     working

     with

     you

    .

     Have

     a

     great

     day

    !

     

    🌟

    ✨

    📝

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     historical

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     Lou

    vre

     Museum

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     It

     is

     also

     famous

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     culture

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     cultural

     center

     for

     France

    .

     It

     is

     home

     to

     many

     important

     institutions

    ,

     including

     the

     É

    cole

     Poly

    techn

    ique

     and

     the

     Lou

    vre

     Museum

    .

     
    


    (Note

    :

     While

     Paris

     is

     the

     capital

     of

     France

    ,

     it

     is

     not

     the

     only

     city

     in

     the

     country

    .

     Other

     major

     cities

     include

     Paris

    ,

     Lyon

    ,

     Marseille

    ,

     and

     N

    antes

    .)

     
    


    (

    3

    6

    0

     words

    )

     France

    's

     capital

     city

    ,

     Paris

    ,

     is

     renowned

     for

     its

     rich

     history

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     diverse

    ,

     with

     many

     different

     areas

     of

     development

     and

     innovation

    .

     Here

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

     future

     of

     AI

    :
    


    1

    .

     Increased

     automation

    :

     As

     automation

     technology

     improves

    ,

     we

     can

     expect

     more

     and

     more

     jobs

     to

     be

     automated

    ,

     reducing

     the

     need

     for

     human

     workers

    .

     This

     could

     lead

     to

     new

     job

     creation

     in

     areas

     like

     robotics

    ,

     data

     analysis

    ,

     and

     software

     development

    .
    


    2

    .

     Deep

     learning

     and

     machine

     learning

    :

     These

     are

     two

     key

     areas

     of

     AI

     that

     are

     rapidly

     evolving

    .

     Deep

     learning

     is

     focused

     on

     creating

     artificial

     neural

     networks

     that

     can

     learn

     and

     make

     decisions

     on

     their

     own

    ,

     while

     machine

     learning

     is

     focused

     on

     creating

     algorithms

     that

     can

     learn

     from

     data

    .
    


    



```python
llm.shutdown()
```
