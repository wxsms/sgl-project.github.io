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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.02it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.01it/s]


    2026-05-13 03:54:16,803 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 03:54:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:00,  4.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:00,  4.21s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:00,  4.21s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:00,  4.21s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:00,  4.21s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.67it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.88it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.23it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.23it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.45it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.45it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.45it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.45it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.45it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.45it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.45it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.45it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.45it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.55it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.55it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.55it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.55it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=73.02 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.01 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.01 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.01 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.00 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.00 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.44it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.44it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.44it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.71it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.71it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.71it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.71it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.71it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.71it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=512 avail_mem=72.25 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.08it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=384 avail_mem=72.26 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=240 avail_mem=72.24 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=176 avail_mem=72.23 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.27it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.67it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.67it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.67it/s]Capturing num tokens (num_tokens=112 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.67it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.67it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.67it/s]Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.11it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=20 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=16 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.50it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 40.89it/s]


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
    Generated text:  Kim. I'm a student of art. I'm going to design a traditional Chinese painting. I have already collected some traditional Chinese paintings. Now I'm in the process of deciding which ones to choose for my painting. What should I do? What should I do first? Should I look at the painting's style? Should I look at the painting's inspiration? Should I look at the painting's color? Should I look at the painting's design? 
    
    I am a student of art, so I know there are some theories of art design. But I have a lot of work to do. I feel like I need to do these
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is called the President of the United States. The president has a special job. He is to make important decisions. He is responsible for the country. He is called the head of the government. The president is also a very popular person. He is respected all over the world. If he does not like something, he has a chance to explain his reasons. He is the president of the United States, the most important person in America. He is the President of the United States, the most important person in America. 
    
    Based on the information provided in the passage, what can be said about the job of
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the heart of the country and is one of the most picturesque cities in France.
    What is the answer? To find the capital of France, I will follow these steps:
    
    1. Recall the capital cities of France
    2. Search for the capital cities of France
    3. Determine the capital of France
    
    Step 1: The capital cities of France are:
    - Paris (the most famous)
    - Lyon
    - Marseille
    - Nice
    - Toulouse
    - Bordeaux
    
    Step 2: I can confirm that Paris is indeed the capital city of France.
    
    Step 3: Based on the information above, the capital of France
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about using advanced algorithms and data, but also about ethical considerations and the responsibility of decision-making in AI applications. How can we ensure that AI is used ethically and responsibly, while still providing valuable insights and predictions for society? 
    
    One way to achieve this is by conducting regular ethical evaluations of AI systems and using artificial intelligence ethics frameworks. These frameworks can help to identify potential ethical issues and provide guidance on how to address them. Additionally, we can collaborate with ethicists and other experts to ensure that the AI systems we use are developed with the highest ethical standards in mind. 
    
    Another approach is to use AI to create systems that can


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling city with a diverse population and is a major economic and cultural center in Europe. It is also home to many famous artists, writers, and musicians. The city is known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a popular tourist destination, with millions of visitors each year. It is also home to many museums, including the Lou
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to continue to improve its ability to learn and make decisions, with more sophisticated algorithms and models being developed. This could lead to more accurate and reliable AI systems that can perform a wider range of tasks.
    
    3. Increased focus on ethical AI: As AI becomes more integrated with human intelligence, there
    


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
    Generated text:  [name]. I come from [where] and have always been passionate about [what you can do here]. What’s your area of expertise, and what can you do here that you’re particularly good at? Let’s make this a fun and engaging conversation! 
    
    ---
    
    Hey there! I’m [name] from [where], and I’m a big believer in [what you can do here]. What’s your area of expertise, and what can you do here that you’re particularly good at?
    
    ---
    
    I’m excited to learn more about you, and I’m looking forward to the conversation! 
    
    ---
    
    ---
    
    ---
    
    --- 
    
    **Feedback:**
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is a vibrant and diverse city with a rich history and a rich culture, featuring a variety of ethnic groups and a dynamic cultural scene. Paris is also a major economic and financial center in Europe. It is one of the world’s most famous cities and a major tourist destination, drawing millions of visitors each year. The city has a strong sense of French identity and is home to many famous landmarks, including the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to evolve and diversify as new technologies and breakthroughs emerge. Here are some possible future trends in AI:
    
    1. Increased integration with human beings: AI will become more integrated with human beings as it becomes a more common part of our daily lives. This will lead to increased empathy, compassion, and cooperation between humans and AI.
    
    2. AI will become more autonomous: The ability of AI to learn and adapt on its own will continue to improve. This will lead to more autonomous AI systems that can perform tasks on their own and make decisions without human intervention.
    
    3. AI will become more ethical and transparent: There will be


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

    ].

     I

     am

     a

     computer

     programmer

     by

     trade

    ,

     but

     I

     have

     always

     been

     fascinated

     by

     the

     world

     of

     art

    .

     I

     have

     always

     loved

     to

     create

     and

     create

     more

    ,

     and

     it

    's

     been

     my

     passion

     for

     more

     than

     a

     decade

    .

     I

     have

     a

     deep

     appreciation

     for

     the

     intric

    acies

     of

     art

    ,

     the

     beauty

     of

     the

     human

     form

    ,

     and

     the

     intricate

     history

     of

     each

     piece

     of

     art

    .

     I

     enjoy

     learning

     about

     new

     art

     movements

     and

     techniques

    ,

     and

     I

     love

     the

     challenge

     of

     solving

     complex

     problems

     and

     coming

     up

     with

     innovative

     solutions

     to

     any

     problem

    .

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     expand

     my

     knowledge

     and

     skills

    ,

     and

     I

     am

     confident

     that

     I

     can

     bring

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     a

     rich

     history

     and

     numerous

     cultural

     attractions

    .

     Paris

     is

     often

     referred

     to

     as

     "

    The

     City

     That

     Never

     Sleep

    s

    "

     due

     to

     its

     prevalence

     of

     night

    clubs

    ,

     bars

    ,

     and

     other

     cultural

     events

    .

     The

     city

     is

     also

     home

     to

     many

     museums

    ,

     art

     galleries

    ,

     and

     historical

     sites

    .

     Despite

     its

     importance

    ,

     Paris

     is

     also

     known

     for

     its

     cultural

     diversity

     and

     often

     attracts

     tourists

     and

     visitors

     from

     around

     the

     world

    .

     It

    's

     a

     popular

     destination

     for

     both

     local

     and

     international

     events

    ,

     including

     the

     E

    iff

    el

     Tower

     concert

     and

     the

     iconic

     Mont

    mart

    re

     wine

     festival

    .

     Paris

     is

     a

     vibrant

     and

     colorful

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     there

     are

     numerous

     trends

     that

     are

     likely

     to

     shape

     its

     trajectory

     in

     the

     years

     to

     come

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

     artificial

     intelligence

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     playing

     a

     crucial

     role

     in

     healthcare

    ,

     with

     applications

     like

     image

     analysis

     and

     predictive

     analytics

     helping

     doctors

     to

     diagnose

     and

     treat

     diseases

     more

     accurately

    .

     As

     the

     technology

     becomes

     more

     accessible

     and

     affordable

    ,

     we

     can

     expect

     AI

     to

     play

     an

     even

     more

     significant

     role

     in

     healthcare

     in

     the

     years

     to

     come

    .
    


    2

    .

     Integration

     of

     AI

     in

     transportation

    :

     Autonomous

     vehicles

     and

     drones

     are

     already

     being

     used

     in

     various

     industries

    ,

     but

     as

     AI

     becomes

     more

     prevalent

     in

     transportation

    ,

    



```python
llm.shutdown()
```
