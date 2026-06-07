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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.48it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.33it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.33it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.44it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.44it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.44it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.44it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.44it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.44it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.44it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.41it/s]Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.41it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.41it/s]Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.51it/s]Capturing num tokens (num_tokens=832 avail_mem=71.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.51it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.51it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.51it/s]Capturing num tokens (num_tokens=640 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.51it/s]Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.51it/s]Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=480 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.38it/s]

    Capturing num tokens (num_tokens=448 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.65it/s]Capturing num tokens (num_tokens=352 avail_mem=71.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.65it/s]Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.65it/s]Capturing num tokens (num_tokens=288 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.65it/s]Capturing num tokens (num_tokens=256 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.65it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.65it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.56it/s]Capturing num tokens (num_tokens=224 avail_mem=71.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.56it/s]Capturing num tokens (num_tokens=208 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.56it/s]

    Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.56it/s]Capturing num tokens (num_tokens=176 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.56it/s]Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.56it/s]Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=144 avail_mem=71.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=96 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s] Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.39it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.39it/s]Capturing num tokens (num_tokens=48 avail_mem=71.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.39it/s]

    Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.39it/s]Capturing num tokens (num_tokens=28 avail_mem=71.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.39it/s]Capturing num tokens (num_tokens=24 avail_mem=71.62 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.39it/s]Capturing num tokens (num_tokens=24 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=12 avail_mem=71.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s] Capturing num tokens (num_tokens=4 avail_mem=71.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 37.58it/s]


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
    Generated text:  Saito Aya and I'm a PhD student at the department of Computational and Mathematical Sciences (CMSC) of the University of Arizona. My main research interests are in the field of mathematics, with a special focus on algebraic geometry, and specifically on Gromov-Witten theory. My research work involves analyzing and computing results on the Gromov-Witten theory, which is a fundamental area in algebraic geometry that studies the geometry of algebraic varieties and their invariants. My research work uses a combination of algebraic geometry techniques and computational methods to analyze these invariants. My main focus is to understand the algebraic and
    ===============================
    Prompt: The president of the United States is
    Generated text:  22 years older than the president of Texas, and the president of Texas is half the age of the president of the United States. If the president of the United States is currently 73 years old, what will be the president of the United States and the president of Texas ages 5 years from now? To solve this problem, we need to determine the current ages of the presidents of the United States and Texas, and then use these values to calculate their ages 5 years from now.
    
    1. **Determine the current age of the president of Texas:**
       - The president of the United States is currently 7
    ===============================
    Prompt: The capital of France is
    Generated text: : ____
    A. Paris
    B. Lyon
    C. Lille
    D. Nice
    Answer: A
    
    Which of the following is a common feature of multiple pneumonias?
    A. Formation of a mass in the chest
    B. Dyspnea
    C. Enlargement of the hilar lymph nodes
    D. Coughing up purulent sputum
    E. Presence of moist rales in the lungs
    Answer: E
    
    Which of the following is a standard test for diagnosing chronic bronchitis?
    A. Pulmonary function test
    B. Chest X-ray
    C. Cardiac examination
    D
    ===============================
    Prompt: The future of AI is
    Generated text:  heading from the sky, not the earth. AI is evolving from its early days in the 1950s to where it is now, and it’s reaching a point of convergence in that, the AI that is being developed right now, it is not capable of replicating everything that we can learn from Earth.
    If we look back at the history of AI and we look at the whole journey, we see a lot of progress, but we also see a lot of setbacks. In the 1950s, the world’s first AI project was to build a computer that could play chess, and we have to be


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Fluviale" or simply "La Ville". It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 5 million people. Paris is known for its rich history, art, and culture, and is a major tourist destination. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the surrounding landscape. The city is also home to many famous French artists, writers, and musicians.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more robust and transparent AI systems that are designed to minimize harm and maximize safety.
    
    3. Increased
    


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
    Generated text:  Emily. I'm a journalist covering the news of the day. Where do you work? I work at the local newspaper. What do you like to do outside of work? I like to read and watch movies. How do you stay healthy? I eat healthy food and exercise regularly. What's your favorite hobby? I love to travel and explore new things. What's your favorite book? "To Kill a Mockingbird" by Harper Lee. What's your favorite color? I like blue. Can you tell me more about yourself? Emily, you are a fascinating character. With your extensive knowledge of the world, it's amazing to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its iconic architecture, vibrant culture, and rich history. Paris has a population of over 2 million people and is a major economic hub, hosting world-renowned museums, fashion houses, and art galleries. It is also famous for its cuisine, music, and fashion.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be highly dynamic and influenced by a variety of factors. Here are some possible trends that could be expected in the field:
    
    1. Increased automation: One of the most significant trends in AI is the increasing automation of tasks that are currently done by humans. This could include tasks such as data analysis, natural language processing, and decision making. As AI technology becomes more advanced and efficient, we may see more automation in industries such as manufacturing, finance, and healthcare.
    
    2. Enhanced human-AI collaboration: Another trend is the increasing collaboration between humans and AI. This could involve more personalized and tailored AI systems that can understand and respond to


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

    'm

     a

    /an

     [

    insert

     profession

     or

     area

     of

     expertise

    ].

     I

     enjoy

     [

    insert

     something

     that

     describes

     your

     hobbies

     or

     interests

    ].

     I

     love

     [

    insert

     something

     that

     describes

     you

     as

     a

     person

    ].

     I

    'm

     always

     looking

     for

     new

     experiences

     and

     learning

     new

     things

    .

     I

    'm

     always

     striving

     to

     improve

     myself

     and

     be

     better

    .

     I

    'm

     here

     to

     stay

    .

     What

     would

     you

     like

     to

     know

     more

     about

     me

    ?

     [

    Make

     your

     introduction

     brief

    ,

     direct

    ,

     and

     engaging

    .

     The

     key

     is

     to

     let

     your

     personality

     and

     interests

     shine

     through

    .]

     Start

     with

     an

     interesting

     opening

     that

     grabs

     the

     reader

    's

     attention

    .

     Be

     concise

     and

     informative

    .

     Avoid

     unnecessary

     j

    argon

     or

     technical

     language

     unless

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     and

     the

     largest

     metropolitan

     area

     in

     Europe

     by

     population

    ,

     and

     the

     third

     largest

     in

     the

     world

    .

     The

     city

     is

     the

     seat

     of

     government

     for

     the

     French

     Republic

     and

     for

     most

     of

     the

     European

     Union

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     art

     and

     culture

    ,

     and

     unique

     architectural

     style

    ,

     which

     is

     the

     reason

     for

     its

     fame

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

    .

     It

     is

     also

     home

     to

     many

     historical

     landmarks

     and

     museums

    ,

     including

     the

     Lou

    vre

     Museum

    ,

     the

     Palace

     of

     Vers

    ailles

    ,

     the

     Tro

    cad

    éro

    ,

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     The

     city

     is

     also

     known

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     highly

     dynamic

    ,

     with

     many

     different

     trends

     and

     developments

     currently

     shaping

     the

     landscape

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

     AI

     will

     become

     more

     accessible

    :

     With

     the

     development

     of

     more

     affordable

     AI

     tools

     and

     platforms

    ,

     we

     can

     expect

     AI

     to

     become

     more

     accessible

     to

     a

     wider

     range

     of

     people

    .

     This

     will

     make

     AI

     even

     more

     usable

     and

     valuable

     in

     many

     applications

    .
    


    2

    .

     AI

     will

     continue

     to

     improve

    :

     As

     we

     continue

     to

     invest

     in

     AI

     research

     and

     development

    ,

     we

     can

     expect

     to

     see

     significant

     improvements

     in

     AI

     algorithms

     and

     models

    .

     This

     will

     likely

     lead

     to

     more

     accurate

     and

     efficient

     AI

     systems

     that

     can

     handle

     a

     wider

     range

     of

     tasks

    .
    


    



```python
llm.shutdown()
```
