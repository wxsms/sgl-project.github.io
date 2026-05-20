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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.33it/s]


    2026-05-20 19:35:50,924 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 19:35:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.68it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.68it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.91it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.91it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.91it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.91it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.91it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.91it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.91it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.91it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.91it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.91it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.91it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.15it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.15it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.15it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.15it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.15it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.15it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.15it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.15it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.15it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.15it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.15it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.16it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.16it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.16it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.16it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.16it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.16it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.16it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.16it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.47it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.20it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.20it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.53it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.53it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.53it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.53it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.53it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.53it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.35it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.46it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.46it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.46it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.46it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.46it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.46it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.02it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.02it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.02it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.85it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 44.89it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.35it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.93it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.93it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 39.93it/s]


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
    Generated text:  Caroline. I am a teacher at a public high school and I have been teaching for 2 years now. I have been trying to come up with a way to have a fun and creative lesson that I can use to teach my students about the intricate world of dinosaurs. The idea is to have a dinosaur themed Halloween in our school. How can I do this and what can I do to make the lesson fun for my students? Can you provide some ideas for a dinosaur-themed Halloween lesson that is interesting and engaging for students? One idea could be to have a scavenger hunt around the school, where students have to find and identify dinosaurs.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a typical democracy, where citizens elect representatives to form a government. What is the function of the president of the United States? A. Representing the state B. Protecting the country C. Managing the country D. Governing the country
    
    Answer:
    D. Governing the country
    
    In a democracy, the president of the United States represents the citizens in the government and has the authority to govern the country. They are responsible for representing the interests of the people and passing laws and policies that benefit the citizens.
    
    A. Representing the state: While the president does represent the state, it is the state that makes the laws, and
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. London
    C. New York
    D. Tokyo
    Answer:
    A
    
    Which of the following statements about China's natural geography is correct? A. It has vast, arid continental terrain. B. It has a wide variety of rivers and lakes. C. It has a large area and many river ports. D. It borders the Pacific Ocean to the east. 
    Answer:
    C
    
    The climate type of the province where the National Museum of China is located is ____
    A. Temperate Oceanic Climate
    B. Temperate Monsoon Climate
    C. Subtropical Monsoon Climate
    ===============================
    Prompt: The future of AI is
    Generated text:  very exciting, and it's not just limited to the technological advancements we've seen in the past decade. AI is being used in all sorts of ways, from medical treatments to autonomous vehicles, and it's only set to continue to grow in the future. However, with such a wide range of applications, it can be difficult to know where to start when it comes to implementing AI systems in your organization. In this article, we'll provide an overview of some of the key considerations to keep in mind when implementing AI systems in your organization.
    AI is a rapidly evolving field, and there are many different approaches to implementing it. Some of the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests]. I like to read, watch movies, and listen to music. What's your favorite hobby? I like to [insert a short description
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French Parliament building. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the Middle Ages. It is a major transportation hub and a major tourist destination. The city is known for its cuisine, fashion, and music, and is home to many famous landmarks and attractions. Paris is a vibrant and dynamic city with a strong sense of French identity and culture. It is a major economic and financial center in Europe and a major player
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more and more AI systems are used in decision-making processes, there is a growing emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and fairness. AI developers will need to prioritize ethical considerations in their work and strive to create AI systems that are fair and unbiased.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, and transportation.
    


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
    Generated text:  [insert fictional character's name], and I am a/an [insert fictional character's profession or role]. My professional title is [insert professional title] and I am currently working as [insert fictional character's job title or role]. My current workplace is [insert fictional workplace's name]. I am a/an [insert fictional character's profession or role], and I'm currently working at [insert fictional workplace's name]. I am [insert fictional character's age or age range] years old, and I'm [insert fictional character's gender] [insert fictional character's gender identifier] [insert fictional character's gender identifier]. I have a broad
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This information can be used to answer the following question: When was the last time Paris was used as the capital of France? Paris was the capital of France from 1804 until 1963. The year 1963 is significant because it marked the end of the monarchy in France, which had ruled since the French Revolution in 1789. The last time Paris was used as the capital of France was in 1963. The last time Paris was used as the capital of France was in 1963. 
    
    Note that the year 196
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and diverse. Here are some possible trends in AI:
    
    1. Improved transparency: AI models are becoming more transparent, allowing users to understand how decisions are made. This is particularly important in industries like healthcare and finance where data sharing is a concern.
    
    2. Enhanced data management: AI will become more adept at managing and analyzing large amounts of data. This will help businesses make more informed decisions, which will in turn lead to better performance.
    
    3. Increased personalization: With the rise of big data, AI will be able to tailor content to individual users, leading to more personalized experiences. This will help businesses stand out in a crowded market


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

     a

     [

    insert

     profession

    ].

     I

    'm

     a

     [

    insert

     personality

     characteristic

    ]

     personality

     type

    ,

     and

     I

     enjoy

     [

    insert

     hobbies

     or

     interests

    ].

     I

    'm

     [

    insert

     age

    ]

     years

     old

    ,

     and

     I

     love

     [

    insert

     something

     related

     to

     my

     interests

     or

     hobbies

    ].

     I

    'm

     [

    insert

     any

     other

     relevant

     information

    ,

     such

     as

     a

     job

     title

    ,

     a

     favorite

     color

    ,

     or

     a

     unique

     skill

    ]

     and

     I

    'm

     [

    insert

     any

     other

     relevant

     information

    ,

     such

     as

     a

     favorite

     book

     or

     a

     funny

     joke

    ].


    So

    ,

     if

     you

    're

     in

     a

     hurry

     or

     just

     want

     to

     hear

     my

     intro

    ,

     I

     can

     tell

     you

     "

    Hello

    ,

     my

     name

     is

     [

    insert

     name

    ],

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     bustling

     met

    ropolis

     known

     for

     its

     rich

     history

    ,

     diverse

     culture

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

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .
    


    Can

     you

     provide

     the

     French

     name

     for

     Paris

     and

     the

     meaning

     of

     the

     city

    's

     name

    ?

     The

     French

     name

     for

     Paris

     is

     "

    Paris

    "

     and

     the

     meaning

     of

     the

     city

     is

     "

    City

     of

     Light

    ."
    


    Can

     you

     give

     me

     a

     list

     of

     famous

     landmarks

     in

     Paris

    ?

     Sure

    ,

     some

     of

     the

     most

     famous

     landmarks

     in

     Paris

     include

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

     the

     Arc

     de

     Tri

    omp

    he

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     full

     of

     exciting

     new

     possibilities

     and

     breakthrough

    s

    .

     Some

     of

     the

     most

     promising

     trends

     include

    :
    


    1

    .

     Increased

     AI

     transparency

     and

     explain

    ability

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     lives

    ,

     we

    'll

     need

     to

     be

     able

     to

     understand

     how

     it

     makes

     decisions

     and

     how

     it

     makes

     mistakes

    .

     This

     is

     where

     AI

     transparency

     and

     explain

    ability

     will

     play

     a

     key

     role

    ,

     making

     it

     easier

     for

     people

     to

     understand

     and

     trust

     the

     algorithms

     and

     models

     that

     power

     our

     technology

    .
    


    2

    .

     Artificial

     Intelligence

     for

     healthcare

    :

     AI

     will

     be

     a

     game

    -ch

    anger

     in

     healthcare

    ,

     with

     the

     ability

     to

     analyze

     vast

     amounts

     of

     data

     to

     improve

     diagnostic

     accuracy

     and

     treatment

     outcomes

    .

     This

     includes

     tools

     like

     AI

    -powered

    



```python
llm.shutdown()
```
