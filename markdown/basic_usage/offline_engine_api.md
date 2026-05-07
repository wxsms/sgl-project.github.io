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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.22it/s]


    2026-05-07 09:17:48,904 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 09:17:48] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.35it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.35it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.35it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.35it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.35it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.35it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.35it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.35it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.35it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.35it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.35it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.47it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.47it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.47it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.47it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:04, 13.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:04, 13.69it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.64 GB):   3%|▎         | 2/58 [00:00<00:04, 13.69it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=69.64 GB):   3%|▎         | 2/58 [00:00<00:04, 13.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.64 GB):   9%|▊         | 5/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.64 GB):   9%|▊         | 5/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.63 GB):   9%|▊         | 5/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.62 GB):   9%|▊         | 5/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.58 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.36it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=69.58 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.58 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.58 GB):  21%|██        | 12/58 [00:00<00:01, 27.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.57 GB):  21%|██        | 12/58 [00:00<00:01, 27.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.57 GB):  21%|██        | 12/58 [00:00<00:01, 27.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.57 GB):  21%|██        | 12/58 [00:00<00:01, 27.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.57 GB):  21%|██        | 12/58 [00:00<00:01, 27.68it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.56 GB):  21%|██        | 12/58 [00:00<00:01, 27.68it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.95it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=69.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=69.53 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.95it/s]Capturing num tokens (num_tokens=960 avail_mem=69.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.95it/s] Capturing num tokens (num_tokens=960 avail_mem=69.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=896 avail_mem=69.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=832 avail_mem=69.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=768 avail_mem=69.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=704 avail_mem=69.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=640 avail_mem=69.53 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=640 avail_mem=69.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=576 avail_mem=69.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=512 avail_mem=69.52 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.09it/s]

    Capturing num tokens (num_tokens=480 avail_mem=69.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=448 avail_mem=69.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=416 avail_mem=69.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=416 avail_mem=69.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.18it/s]Capturing num tokens (num_tokens=384 avail_mem=69.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.18it/s]Capturing num tokens (num_tokens=352 avail_mem=69.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.18it/s]Capturing num tokens (num_tokens=320 avail_mem=69.51 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.18it/s]Capturing num tokens (num_tokens=288 avail_mem=69.51 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.18it/s]Capturing num tokens (num_tokens=256 avail_mem=69.51 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.18it/s]Capturing num tokens (num_tokens=256 avail_mem=69.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.38it/s]Capturing num tokens (num_tokens=240 avail_mem=69.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.38it/s]Capturing num tokens (num_tokens=224 avail_mem=69.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.38it/s]

    Capturing num tokens (num_tokens=208 avail_mem=69.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.38it/s]Capturing num tokens (num_tokens=192 avail_mem=69.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.38it/s]Capturing num tokens (num_tokens=176 avail_mem=69.49 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.38it/s]Capturing num tokens (num_tokens=176 avail_mem=69.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=160 avail_mem=69.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=144 avail_mem=69.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=128 avail_mem=69.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=112 avail_mem=69.48 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=96 avail_mem=69.48 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.10it/s] Capturing num tokens (num_tokens=96 avail_mem=69.48 GB):  81%|████████  | 47/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=80 avail_mem=69.48 GB):  81%|████████  | 47/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=64 avail_mem=69.47 GB):  81%|████████  | 47/58 [00:01<00:00, 44.52it/s]

    Capturing num tokens (num_tokens=48 avail_mem=69.47 GB):  81%|████████  | 47/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=32 avail_mem=69.47 GB):  81%|████████  | 47/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=28 avail_mem=69.46 GB):  81%|████████  | 47/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=28 avail_mem=69.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=24 avail_mem=69.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=20 avail_mem=69.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=16 avail_mem=69.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=12 avail_mem=69.45 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=8 avail_mem=69.45 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.17it/s] Capturing num tokens (num_tokens=8 avail_mem=69.45 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=4 avail_mem=69.44 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=4 avail_mem=69.44 GB): 100%|██████████| 58/58 [00:01<00:00, 39.25it/s]


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
    Generated text:  Nicolette and I am a research scientist at the University of Copenhagen. My area of expertise is on the biochemistry of the human immune system. I currently work on several projects. One of them, the current RIA Project, focuses on the role of T cells in the development and maintenance of B cell cutaneous lymphomas. This research is important because understanding the role of T cells in B cell cutaneous lymphomas will allow us to understand the mechanisms that lead to this type of lymphoma and identify potential therapeutic targets.
    I was previously a postdoctoral researcher at the University of Copenhagen, where I worked on the gut microbiome and
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy, so he often works with a secretary. In 2017, he worked with the secretary for 12 days. How many days were there in total in that year?
    
    To determine the total number of days in 2017 that the president of the United States worked with the secretary, we need to consider the following:
    
    1. The president worked for 12 days.
    2. The secretary worked for the entire year of 2017, which is 365 days.
    
    We can find the total number of days by adding the number of days the president worked with the number of
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. In this city, there is a well-known cafe called Le Bar, a popular place where people can relax and enjoy a cup of coffee. In order to find the capital of France, I need to know what the capital of France is. After some research, I found that Paris is the capital of France. Can you tell me where the capital of France is located? Based on the information provided, the capital of France is Paris. The capital of France is located in the south of the country, specifically in the region of the Loire Valley. The Loire Valley is a scenic region with beautiful mountains, forests, and rivers
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, with many researchers and engineers working hard to improve our understanding of the technology and its potential applications. One of the most promising areas of AI research is in the development of more advanced algorithms and models. These algorithms and models are designed to be more accurate, faster, and more adaptable than those currently in use. This not only improves the performance of existing systems but also opens up new possibilities for developing new applications and technologies.
    There are several ways in which researchers and engineers can improve the performance of AI algorithms and models. One approach is to apply machine learning techniques, which involve training a model on a large dataset to identify patterns and relationships in


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have a [job title] at [company name]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm always looking for new challenges and opportunities to grow and learn. What do you enjoy doing? I enjoy reading, traveling, and spending time with my family. What do you like to do in your free time? I like to read
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of French literature, art, and music, and is a major economic and cultural center. Paris is home to many world-renowned museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. The city is also known for its rich history, including the French Revolution and the French Revolution Monument. Paris is a popular tourist destination, with millions of visitors annually. The city is also home to many cultural institutions, including the Mus
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced ethical considerations: As AI becomes more advanced, there will be increased scrutiny of its ethical implications, including issues of bias, privacy, and accountability.
    
    3. Greater reliance on machine learning: AI will become more prevalent in various industries, with machine learning algorithms becoming more sophisticated and capable of performing tasks that were previously done by humans.
    
    4. Increased focus on AI ethics: There will be increased focus on ethical considerations in
    


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
    Generated text:  [Name] and I am a [age] year old [occupation] (e.g., teacher, software engineer, journalist, etc.). I have always been passionate about [field of interest], which has driven me to achieve great things in my career. Despite facing many challenges and obstacles, I have never given up on my dreams, and I am determined to achieve my goals. I am always looking for new opportunities to learn and grow, and I am always eager to challenge myself and grow as a person. Thank you for taking the time to meet me! 📚✨
    
    Hey there, [Name]. What inspired you to become
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and the second most populous city in the European Union. Paris is also the seat of the French government and government headquarters. It is home to many world-renowned cultural and artistic institutions, including the Louvre Museum, the Eiffel Tower, the Arc de Triomphe, and the Palace of Versailles. The city is also known for its gastronomic and culinary heritage, particularly the Parisian cuisine. Paris has a long and rich history, dating back over 2,000 years. The city is known for its cultural richness, including its museums, galleries, and theaters
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and promising, with several trends that are expected to shape its trajectory. Here are some potential future trends in AI:
    
    1. Increased use of AI in healthcare: AI is expected to play an increasingly important role in healthcare by automating tasks that are time-consuming or expensive, such as diagnosing diseases and managing patient records. Additionally, AI-powered diagnostic tools and predictive analytics can help healthcare providers make more accurate and timely diagnoses, ultimately leading to better patient outcomes.
    
    2. AI in finance: AI is expected to have a significant impact on the finance industry, particularly in areas such as fraud detection, risk management, and investment analytics. AI-powered


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

     character

    's

     name

     here

    ],

     and

     I

     am

     an

     [

    insert

     age

     here

    ]

     year

     old

     [

    insert

     occupation

     here

    ].

     I

     currently

     live

     in

     [

    insert

     location

     here

    ]

     and

     I

     love

     [

    insert

     something

     you

     enjoy

     doing

     here

    ].

     I

     am

     currently

     [

    insert

     a

     character

     trait

     or

     personality

     trait

     here

    ]

     that

     I

     like

     to

     [

    insert

     a

     short

     sentence

     here

    ].

     
    


    I

    'm

     a

     [

    insert

     a

     role

     here

    ],

     and

     I

     love

     [

    insert

     something

     I

     enjoy

     doing

     here

    ]

     and

     [

    insert

     a

     character

     trait

     here

    ].

     I

     am

     also

     a

     [

    insert

     a

     character

     trait

     here

    ]

     that

     I

     like

     to

     [

    insert

     a

     short

     sentence

     here

    ].

     
    


    I

    'm

     very

     [

    insert

     a

     personality

     trait

     here

    
    
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

     is

     known

     for

     its

     rich

     cultural

     history

    ,

     stunning

     architecture

    ,

     and

     vibrant

     nightlife

    .

     The

     city

     is

     home

     to

     the

     Lou

    vre

     Museum

     and

     Notre

    -D

    ame

     Cathedral

    ,

     and

     is

     a

     popular

     tourist

     destination

    .

     Other

     notable

     landmarks

     include

     the

     E

    iff

    el

     Tower

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     as

     it

     is

     considered

     the

     “

    Fashion

     Capital

     of

     the

     World

    .”

     The

     city

     is

     also

     home

     to

     many

     art

     institutions

     and

     museums

    ,

     including

     the

     Mus

    ée

     d

    '

    Or

    say

     and

     the

     Mus

    ée

     de

     l

    '

    Or

    anger

    ie

    .

     Overall

    ,

     Paris

     is

     a

     city

     that

     is

     known

     for

     its

     exceptional

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     the

     rapid

     development

     of

     new

     technologies

    ,

     innovations

    ,

     and

     applications

    .

     Some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     privacy

     and

     ethics

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     an

     increased

     focus

     on

     privacy

     and

     ethical

     considerations

    .

     Businesses

    ,

     governments

    ,

     and

     individuals

     will

     need

     to

     be

     more

     transparent

     about

     how

     AI

     is

     being

     used

     and

     ensure

     that

     data

     is

     being

     used

     responsibly

    .
    


    2

    .

     AI

     development

     towards

     human

    -level

     capabilities

    :

     Future

     AI

     systems

     are

     likely

     to

     be

     designed

     to

     be

     capable

     of

     performing

     tasks

     that

     were

     previously

     considered

     impossible

     for

     humans

    ,

     such

     as

     playing

     high

    -level

     games

    ,

     understanding

     natural

     language

    ,

     and

    



```python
llm.shutdown()
```
