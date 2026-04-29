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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.18it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.17it/s]


    2026-04-29 13:44:09,775 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 13:44:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.20it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.46it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.46it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.46it/s]Compiling num tokens (num_tokens=416):  38%|███▊      | 22/58 [00:05<00:03,  9.46it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=176):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=160):  55%|█████▌    | 32/58 [00:05<00:01, 15.96it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s]

    Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s]Compiling num tokens (num_tokens=24):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s]Compiling num tokens (num_tokens=20):  74%|███████▍  | 43/58 [00:05<00:00, 24.60it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 34.41it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 34.41it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 34.41it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 34.41it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 34.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.09 GB):   3%|▎         | 2/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.08 GB):   3%|▎         | 2/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.08 GB):   3%|▎         | 2/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.08 GB):   3%|▎         | 2/58 [00:00<00:02, 19.57it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.06 GB):   9%|▊         | 5/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.06 GB):   9%|▊         | 5/58 [00:00<00:02, 22.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.83it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=61.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.03 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.01 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=960 avail_mem=61.02 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.18it/s] Capturing num tokens (num_tokens=896 avail_mem=61.02 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=832 avail_mem=61.02 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.18it/s]

    Capturing num tokens (num_tokens=768 avail_mem=61.01 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=704 avail_mem=61.01 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=704 avail_mem=61.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=640 avail_mem=61.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=576 avail_mem=61.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=512 avail_mem=60.99 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=480 avail_mem=61.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=448 avail_mem=61.00 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=416 avail_mem=61.00 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=416 avail_mem=61.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.99it/s]Capturing num tokens (num_tokens=384 avail_mem=61.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.99it/s]Capturing num tokens (num_tokens=352 avail_mem=61.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.99it/s]Capturing num tokens (num_tokens=320 avail_mem=60.99 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.99it/s]

    Capturing num tokens (num_tokens=288 avail_mem=60.99 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.99it/s]Capturing num tokens (num_tokens=256 avail_mem=60.99 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.99it/s]Capturing num tokens (num_tokens=256 avail_mem=60.99 GB):  64%|██████▍   | 37/58 [00:00<00:00, 46.65it/s]Capturing num tokens (num_tokens=240 avail_mem=60.98 GB):  64%|██████▍   | 37/58 [00:00<00:00, 46.65it/s]Capturing num tokens (num_tokens=224 avail_mem=60.98 GB):  64%|██████▍   | 37/58 [00:00<00:00, 46.65it/s]Capturing num tokens (num_tokens=208 avail_mem=60.97 GB):  64%|██████▍   | 37/58 [00:00<00:00, 46.65it/s]Capturing num tokens (num_tokens=192 avail_mem=60.97 GB):  64%|██████▍   | 37/58 [00:00<00:00, 46.65it/s]Capturing num tokens (num_tokens=176 avail_mem=60.97 GB):  64%|██████▍   | 37/58 [00:00<00:00, 46.65it/s]Capturing num tokens (num_tokens=160 avail_mem=60.97 GB):  64%|██████▍   | 37/58 [00:01<00:00, 46.65it/s]Capturing num tokens (num_tokens=160 avail_mem=60.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=144 avail_mem=60.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=128 avail_mem=60.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=112 avail_mem=60.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.39it/s]

    Capturing num tokens (num_tokens=96 avail_mem=60.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.39it/s] Capturing num tokens (num_tokens=80 avail_mem=60.95 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=64 avail_mem=60.95 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=64 avail_mem=60.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.96it/s]Capturing num tokens (num_tokens=48 avail_mem=60.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.96it/s]Capturing num tokens (num_tokens=32 avail_mem=60.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.96it/s]Capturing num tokens (num_tokens=28 avail_mem=60.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.96it/s]Capturing num tokens (num_tokens=24 avail_mem=60.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.96it/s]Capturing num tokens (num_tokens=20 avail_mem=60.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.96it/s]Capturing num tokens (num_tokens=16 avail_mem=60.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.96it/s]Capturing num tokens (num_tokens=16 avail_mem=60.93 GB):  95%|█████████▍| 55/58 [00:01<00:00, 49.63it/s]Capturing num tokens (num_tokens=12 avail_mem=60.92 GB):  95%|█████████▍| 55/58 [00:01<00:00, 49.63it/s]Capturing num tokens (num_tokens=8 avail_mem=60.92 GB):  95%|█████████▍| 55/58 [00:01<00:00, 49.63it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=60.92 GB):  95%|█████████▍| 55/58 [00:01<00:00, 49.63it/s]Capturing num tokens (num_tokens=4 avail_mem=60.92 GB): 100%|██████████| 58/58 [00:01<00:00, 43.84it/s]


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
    Generated text:  Zhang, and I'm studying 2D game development at a university. I'd like to know more about creating a 2D game and game development tools. Can you provide me with some guidance? Sure, I can help you with that! Creating a 2D game and game development tools involves several key steps. First, you need to decide on the game concept, level design, and character designs. Next, you'll need to choose a game development framework or engine, such as Unity or Unreal Engine. Then, you'll need to set up your development environment, including your development tools, environment, and programming languages. Once
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to continue with the current tax rates or to propose a new tax rate. He has a total of 10 million people in the country and wants to ensure that everyone pays the same amount of tax. The tax rates for the current year are $0.90 per $100 of taxable income. 
    
    The president decides to propose a new tax rate of $0.95 per $100 of taxable income. What will be the tax paid by each person under the new tax rate, and how much more will each person pay in total if the new tax rate is implemented?
    To determine the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the most famous city in France. Paris is the capital of France. The capital of France is Paris. Paris is the most famous city in France. Paris is the capital of France. Paris is the most famous city in France. Paris is the capital of France. Paris is the most famous city in France. Paris is the capital of France. Paris is the most famous city in France. Paris is the capital of France. Paris is the most famous city in France. Paris is the capital of France. Paris is the most famous city in France. Paris is the capital of France. Paris is the most famous city in
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of users
    A. True
    B. False
    Answer:
    
    A
    
    When selecting artificial intelligence tools, what should the user consider?
    A. Characteristics of the system
    B. What the system is used for
    C. Speed of the system
    D. Cost of the system
    Answer:
    
    B
    
    A. Correct
    B. Incorrect
    C. Cannot be determined
    D. Uncertain
    Which of the following is correct regarding the characteristics of an artificial intelligence tool?
    Answer:
    
    A
    
    Please answer the following question based on the given options. Which of the following is a key principle of the principal-agent relationship?
    A


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


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is located on the Seine River and is the seat of government for the country. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its fashion industry, with many famous fashion designers and boutiques. The city is a cultural hub and is home to many museums, theaters, and other cultural institutions. Paris is a major transportation hub
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI continues to advance, we can expect to see more automation and artificial intelligence in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for new jobs.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need for privacy and security. This could lead to new regulations and standards for AI development and
    


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
    Generated text:  Jane and I'm a self-proclaimed living legend. I'm a young woman with an exceptional talent for creating engaging and entertaining storytelling. I've always been a storyteller from the moment I was a little girl, and now I've grown up to be a master storyteller of tales that are both entertaining and profound. I believe in the power of words to transport people to different worlds, and I'm always eager to share my insights on the art of storytelling with others. I'm an advocate for the power of language and believe that stories are the greatest gift we have. I'm confident in my abilities and I'm ready to take on
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital of France and the largest city in the European Union. It is also the world's 6th-largest city by population. It is known for its iconic Eiffel Tower, Notre-Dame Cathedral, and many other landmarks. The city is known for its rich history and culture, including the French Revolution and the Opéra Garnier. Paris is also famous for its fashion and art scene, including the Louvre Museum and the Musée d'Orsay. The city is home to numerous museums, theaters, and art galleries, and has become one of the most visited cities in the world. 
    
    Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising and will continue to evolve rapidly. Here are some possible trends that AI is likely to experience in the coming years:
    
    1. Increased accuracy and precision: One of the key areas where AI is likely to see significant improvement is in areas such as image and speech recognition, natural language processing, and predictive analytics. With continued advancements in machine learning and deep learning, these technologies will become even more accurate and precise, allowing for more accurate diagnoses, better recommendations, and improved customer service.
    
    2. Enhanced cognitive abilities: AI will continue to evolve beyond its basic computational capabilities and will become more capable of understanding and interpreting human emotions, cultural biases,


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

    'm

     a

     [

    job

     title

    ]

     with

     over

     [

    number

    ]

     years

     of

     experience

     in

     [

    industry

    ].

     My

     unique

     selling

     point

     is

     [

    what

     makes

     me

     stand

     out

    ].

     What

     is

     your

     background

     and

     how

     have

     you

     developed

     your

     skills

     and

     knowledge

     in

     [

    industry

    ]

    ?


    I

     look

     forward

     to

     the

     opportunity

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     [

    Your

     Name

    ]

     


    I

    'm

     excited

     to

     meet

     you

     and

     start

     a

     conversation

     about

     what

     makes

     you

     special

    .

     What

    's

     the

     best

     part

     of

     being

     a

     [

    job

     title

    ]

    ?


    I

     have

     always

     been

     fascinated

     by

     the

     world

     of

     [

    industry

    ]

     and

     I

     love

     the

     challenge

     of

     trying

     to

     solve

     complex

     problems

    .

     What

     makes

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     southeastern

     region

     of

     the

     country

    .
    


    How

     would

     you

     explain

     the

     distinction

     between

     public

     and

     private

     schools

     in

     terms

     of

     the

     type

     of

     educational

     and

     social

     services

     provided

    ?

     Explain

     with

     example

    .

     The

     distinction

     between

     public

     and

     private

     schools

     in

     France

     is

     primarily

     based

     on

     the

     type

     of

     educational

     and

     social

     services

     provided

     to

     students

    .

     Public

     schools

    ,

     by

     definition

    ,

     provide

     a

     comprehensive

     range

     of

     educational

     services

     that

     include

     everything

     from

     traditional

     classes

     to

     vocational

     training

     and

     additional

     support

     for

     students

     with

     special

     needs

    .

     Private

     schools

    ,

     on

     the

     other

     hand

    ,

     offer

     fewer

     comprehensive

     services

     but

     are

     more

     focused

     on

     providing

     individual

    ized

     support

     to

     students

    .
    


    A

     good

     example

     of

     a

     public

     school

     in

     France

     is

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     likely

     to

     include

     some

     of

     the

     following

     trends

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     With

     the

     help

     of

     AI

    ,

     we

     will

     be

     able

     to

     create

     personalized

     experiences

     for

     each

     individual

     user

    .

     For

     example

    ,

     chat

    bots

     and

     virtual

     assistants

     will

     be

     able

     to

     understand

     users

    '

     needs

     and

     preferences

     and

     provide

     tailored

     responses

    .
    


    2

    .

     Autonomous

     Vehicles

    :

     AI

    -powered

     self

    -driving

     cars

     will

     become

     more

     common

    ,

     making

     transportation

     much

     safer

     and

     more

     efficient

    .
    


    3

    .

     Medical

     Diagnosis

     and

     Treatment

    :

     AI

     will

     be

     used

     to

     help

     diagnose

     and

     treat

     diseases

     more

     accurately

    ,

     saving

     lives

     and

     reducing

     costs

    .
    


    4

    .

     AI

     for

     Environmental

     Conservation

    :

     AI

     will

     help

     in

     monitoring

     and

     protecting

     the

     environment

    ,

     preventing

     pollution

     and

    



```python
llm.shutdown()
```
