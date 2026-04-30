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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.35it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.34it/s]


    2026-04-30 16:08:55,770 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 16:08:55] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:15,  3.16it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:15,  3.16it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:15,  3.16it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:15,  3.16it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:15,  3.16it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:15,  3.16it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:15,  3.16it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:15,  3.16it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:04<00:15,  3.16it/s]Compiling num tokens (num_tokens=1792):  16%|█▌        | 9/58 [00:04<00:15,  3.16it/s]Compiling num tokens (num_tokens=1536):  16%|█▌        | 9/58 [00:04<00:15,  3.16it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.55it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:04,  8.55it/s]

    Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:04<00:04,  8.55it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=240):  48%|████▊     | 28/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s]

    Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s]Compiling num tokens (num_tokens=64):  66%|██████▌   | 38/58 [00:05<00:00, 21.50it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 31.34it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 31.34it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 31.34it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 31.34it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 31.34it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 31.34it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 31.34it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 31.34it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 31.34it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 31.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.61it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.61it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.61it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.95it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.95it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.95it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.95it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.00it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.35it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.35it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.35it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.35it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.35it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.35it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.35it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.20it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 47.06it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.19it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.19it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.19it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.19it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.19it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 48.19it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.47it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.47it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.47it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.47it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.47it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.47it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.76it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.76it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.76it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.76it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 41.89it/s]


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
    Generated text:  Jao Sung from South Korea. I have a job offer from the United States, and I want to apply for it. How can I contact my employer, and what are the steps to follow?
    
    For example, I want to contact my employer and ask if they are hiring for a new position, and I want to provide them with my resume and cover letter. I will also want to provide them with my social security number and information on my passport, and I would like to request a confidential and legally binding interview to know more about the position. Lastly, I would like to request a copy of my resume and cover letter, as well
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military tanks to buy. The tank weighs 1200 pounds and the price of a tank is $20,000. The cost of maintenance for each year of an army's service is $100,000. The president estimates that the total cost of having an army for 30 years will be $400,000,000. If the president wants to maximize the number of tanks he can buy, what is the optimal number of tanks he should purchase? The optimal number of tanks the president should purchase is 50. 
    
    Here
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    The capital of France is Paris. Paris is the capital of France. 
    
    To confirm this, let's break down the reasoning:
    
    1. Identify the capital of France: The capital of France is Paris.
    2. Confirm the answer: The capital of France is indeed Paris.
    
    The capital city of France is Paris, which is the most populous and largest city in Europe, located on the left bank of the Seine river in the Île de la Cité. Paris is known for its historical significance, rich culture, and annual festivals. The city is renowned for its art, architecture, cuisine, and fashion, and it is a
    ===============================
    Prompt: The future of AI is
    Generated text:  not only about making computers smarter, but about making them more human-like. These exciting advancements are likely to become the norm in the coming decades. With the help of AI, we can create intelligent machines that can learn, reason, and make decisions based on the data we provide. They can even learn and adapt to our needs, and respond to our questions and requests. Therefore, the future of AI is incredibly exciting, and we can expect to see many more intelligent machines in the years to come. With a growing focus on AI, it's important to prioritize ethical considerations and ensure that AI is used responsibly and ethically. This way,


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Character] who has always been [What motivates you to be who you are]. I am passionate about [What you enjoy doing]. I am a [What you are good at] who is always [What you are good at doing]. I am [What you are good at doing] and I am always [What you are good at doing]. I am [What you are good at doing] and I am always [What you are good at doing]. I am [What you are good at doing] and I am always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major economic center and a major tourist destination. The city is home to many cultural institutions and events, including the Opéra Garnier and the Musée d'Orsay. Paris is a city of contrasts, with its modern and historic aspects blending together to create a unique and fascinating place to live and visit. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into many aspects of our lives, from self-driving cars to personalized medicine. As the technology continues to advance, we can expect to see even more integration into our daily routines, such as voice assistants, virtual assistants, and smart home devices.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated into our lives, there will be a greater emphasis on
    


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
    Generated text:  [Name] and I'm a [job title or role] at [company name]. I'm currently [current job status] at [company name], and my [short bio or achievements] has helped me develop my skills and knowledge in [relevant field or area]. I'm excited to learn more about [company name] and see what opportunities might arise for me. Let's get to know each other! [Name], please! You're a great fit for our team. What brings you to [company name] and what makes you a good fit for this position? [Name], thank you for having me! I'm thrilled to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France and is the largest city in the country. It is the seat of government, the seat of administration, and the cultural, political, and economic center of France. It is also the largest city in the European Union and the most populous city in the world by population. Paris is known for its beautiful architecture, vibrant culture, and cultural institutions such as the Louvre Museum. It is also famous for its Eiffel Tower, the Notre-Dame Cathedral, and the Rive Droite and Rive Left sections of Paris. The city has a rich history and is home to many important landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and diverse, with many potential areas of development and growth. Here are some possible trends in AI that are currently being explored and anticipated:
    
    1. Integration of AI with existing technologies: AI is already being integrated into many existing technologies, such as smartphones, cars, and IoT devices. This trend is expected to continue, with more integration and integration of AI with other emerging technologies, such as quantum computing and blockchain.
    
    2. Emphasis on ethical and responsible AI: As AI becomes more sophisticated, there will be an increasing demand for ethical guidelines and regulations to ensure that AI is used responsibly and beneficially for society. This trend is likely to


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

    'm

     a

     [

    occupation

    ]

     who

     has

     been

     [

    number

     of

     years

     since

     becoming

     an

     adult

    ].

     I

     love

     [

    occupation

    ]

     and

     I

     really

     enjoy

     [

    occupation

    ].

     I

     love

     spending

     time

     with

     [

    people

    ]

     and

     I

     have

     a

     lot

     of

     [

    interest

    s

     or

     hobbies

    ].

     What

     brings

     you

     to

     this

     character

    's

     world

    ,

     and

     what

     makes

     you

     unique

     to

     it

    ?

     (

    Feel

     free

     to

     add

     any

     other

     relevant

     information

     about

     your

     personality

    ,

     interests

    ,

     or

     skills

     that

     you

    'd

     like

     to

     share

    .)

     This

     character

     is

     my

     [

    occupation

    ],

     and

     I

    'm

     [

    occupation

    ].

     I

    've

     been

     in

     this

     industry

     for

     [

    number

     of

     years

    ],

     and

     I

    'm

     really

     passionate

     about

     my

     work

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     is

     known

     for

     its

     rich

     history

     and

     beautiful

     architecture

    .

     It

    's

     home

     to

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

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

     Mont

    mart

    re

    .

     Paris

     is

     also

     a

     cosm

    opolitan

     city

     known

     for

     its

     French

     culture

    ,

     fashion

    ,

     and

     world

    -class

     restaurants

    .

     The

     city

     is

     a

     major

     cultural

     hub

    ,

     known

     for

     its

     art

    ,

     music

    ,

     and

     theater

     scenes

    ,

     and

     is

     considered

     one

     of

     the

     most

     beautiful

     cities

     in

     the

     world

    .

     Its

     status

     as

     a

     global

     capital

     city

     has

     made

     it

     a

     world

    -ren

    owned

     destination

     for

     tourists

     and

     locals

     alike

    .

     Paris

     is

     a

     unique

     and

     fascinating

     city

     with

     its

     own

     distinct

     character

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     trends

    ,

     including

    :
    


    1

    .

     Increased

     integration

     with

     human

     decision

    -making

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     are

     likely

     to

     be

     integrated

     with

     human

     decision

    -making

     processes

    ,

     enabling

     them

     to

     make

     better

    -in

    formed

     decisions

     under

     uncertain

     conditions

    .
    


    2

    .

     Personal

    ization

     and

     automation

    :

     AI

     is

     likely

     to

     become

     even

     more

     personal

     in

     the

     future

    ,

     with

     personalized

     recommendations

     and

     automation

     of

     tasks

     that

     previously

     required

     human

     intervention

    .
    


    3

    .

     Enhanced

     transparency

     and

     accountability

    :

     AI

     systems

     are

     becoming

     more

     transparent

     and

     accountable

    ,

     with

     greater

     transparency

     in

     their

     decision

    -making

     processes

     and

     greater

     accountability

     for

     their

     impact

     on

     society

    .
    


    4

    .

     D

    ivers

    ification

     of

     AI

     applications

    :

     AI

     is

     likely

     to

    



```python
llm.shutdown()
```
