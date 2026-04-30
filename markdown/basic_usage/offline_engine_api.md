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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.10it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.09it/s]


    2026-04-30 11:56:19,413 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 11:56:19] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.11it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.26it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.26it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.26it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.26it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.26it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.26it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.26it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.26it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.26it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.26it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 23.65it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 23.65it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 23.65it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 23.65it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 23.65it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 23.65it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 23.65it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 23.65it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 23.65it/s]

    Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 23.65it/s]Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:05<00:00, 23.65it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 32.45it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 32.45it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 32.45it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 32.45it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 32.45it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 32.45it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 32.45it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.63 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.49it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.49it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.49it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=53.59 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.59 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.59 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.58 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.36it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.36it/s]Capturing num tokens (num_tokens=960 avail_mem=53.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.36it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=53.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=896 avail_mem=53.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=832 avail_mem=53.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=768 avail_mem=53.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=704 avail_mem=53.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=640 avail_mem=53.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=640 avail_mem=53.55 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=576 avail_mem=53.55 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=512 avail_mem=53.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=480 avail_mem=53.55 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=448 avail_mem=53.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=416 avail_mem=53.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.08it/s]

    Capturing num tokens (num_tokens=416 avail_mem=53.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.34it/s]Capturing num tokens (num_tokens=384 avail_mem=53.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.34it/s]Capturing num tokens (num_tokens=352 avail_mem=53.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.34it/s]Capturing num tokens (num_tokens=320 avail_mem=53.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.34it/s]Capturing num tokens (num_tokens=288 avail_mem=53.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.34it/s]Capturing num tokens (num_tokens=256 avail_mem=53.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.34it/s]Capturing num tokens (num_tokens=256 avail_mem=53.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=240 avail_mem=53.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=224 avail_mem=53.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.10it/s]

    Capturing num tokens (num_tokens=208 avail_mem=53.49 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=192 avail_mem=53.49 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=176 avail_mem=53.49 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=176 avail_mem=53.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.15it/s]Capturing num tokens (num_tokens=160 avail_mem=53.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.15it/s]Capturing num tokens (num_tokens=144 avail_mem=53.48 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.15it/s]Capturing num tokens (num_tokens=128 avail_mem=53.45 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.15it/s]

    Capturing num tokens (num_tokens=112 avail_mem=53.45 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.15it/s]Capturing num tokens (num_tokens=112 avail_mem=53.45 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.85it/s]Capturing num tokens (num_tokens=96 avail_mem=53.45 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.85it/s] Capturing num tokens (num_tokens=80 avail_mem=53.44 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.85it/s]Capturing num tokens (num_tokens=64 avail_mem=53.44 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.85it/s]Capturing num tokens (num_tokens=48 avail_mem=53.43 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.85it/s]Capturing num tokens (num_tokens=48 avail_mem=53.43 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=32 avail_mem=53.43 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.14it/s]

    Capturing num tokens (num_tokens=28 avail_mem=53.43 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=24 avail_mem=53.42 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=20 avail_mem=53.42 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=20 avail_mem=53.42 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=16 avail_mem=53.42 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=12 avail_mem=53.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=8 avail_mem=53.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.04it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=53.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=4 avail_mem=53.41 GB): 100%|██████████| 58/58 [00:01<00:00, 31.13it/s]Capturing num tokens (num_tokens=4 avail_mem=53.41 GB): 100%|██████████| 58/58 [00:01<00:00, 33.85it/s]


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
    Generated text:  Emily. I'm a girl who likes to listen to music and spend time with people. I like spending time with my family and friends. I like to play computer games and I like to take walks in the park. I want to be a doctor. 
    
    Is that right? 
    
    I like to listen to music and spend time with people. I like spending time with my family and friends. I like to play computer games and I like to take walks in the park. I want to be a doctor. Does that sound right? 
    
    Why are you different from Emily? Why are you not a doctor? 
    
    I'm not sure.
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to finalize a deal with the president of China. At first, the president of the United States thinks that the president of China is more likely to support the deal than the president of China does. After some negotiations, they agree to the deal. After this, the president of the United States thinks that the president of China is less likely to support the deal than the president of China does. What might be the problem? The most likely problem is that:
    A) The president of the United States and the president of China may have different priorities and priorities when it comes to the deal.
    B) The president of the United States may have
    ===============================
    Prompt: The capital of France is
    Generated text:  _____.
    A. Paris
    B. Lyon
    C. Geneva
    D. Avignon
    Answer:
    A
    
    The currency of Japan is ____.
    A. Japanese Yen
    B. US Dollar
    C. Euro
    D. Swiss Franc
    Answer:
    A
    
    In Japan, the 'god' of the shrine is the ____
    A. Emperor
    B. Shinto God
    C. Buddhist God
    D. Chinese God
    Answer:
    B
    
    Which of the following is not a high-end group of people in Japan?
    A. The Toyota family
    B. The Honda family
    C. The Samsung family
    D. The
    ===============================
    Prompt: The future of AI is
    Generated text:  fast approaching, and we're already seeing some amazing applications. But what about the future of AI in the healthcare industry? We've been hearing about the importance of AI in healthcare and how it can improve patient outcomes. But is it possible to predict the future of AI in the healthcare industry, or is it just a matter of time?
    The answer to this question is a mix of predictability and unpredictability. AI has the potential to revolutionize healthcare in many ways, but it's also subject to rapid change and innovation.
    One way AI can revolutionize healthcare is through the use of machine learning and predictive modeling. This technology can analyze large


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your job or profession]. I enjoy [insert a short description of your hobbies or interests]. What brings you to [company name] and what makes you a good fit for the position? I'm excited to learn more about your company and how I can contribute to your success. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests]. What brings you to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Fluviale" (The River City). It is the largest city in France and the second-largest city in the European Union. Paris is a historic and cultural center with many landmarks, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major financial center and a major transportation hub. Paris is known for its cuisine, fashion, and art, and is a popular tourist destination. It is also home to the French Parliament and the French Academy of Sciences. The city is also home to many international organizations and institutions,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will likely lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve diagnosis and treatment outcomes. As AI continues to advance, we can expect to see even more sophisticated applications in this field, such as personalized medicine
    


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
    Generated text:  [Name] and I'm a [Job Title] with [Company Name]. I'm passionate about [Your Passion], and I love to [Your Current Activity]. [Name] is excited to chat with you about [Your Topic of Interest]. Let's connect! 📈👋
    
    ---
    
    I'd love to hear your interesting story! Can you share a bit about yourself? 🤔
    
    ---
    
    Tell us more about what you're passionate about and how you got into the field of [Your Passion]. 🤔
    
    ---
    
    What inspired you to become a [Your Job Title] and what do you like to do for fun?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "la Pop," a historic city on the Seine River, home to UNESCO World Heritage sites such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Museum. Paris is known for its artistic, cultural, and historical significance and is considered the cultural and economic center of the European Union. The city is also home to numerous international festivals, such as the Eiffel Tower Festival and the Marais Art Walking Tour. Paris has a rich cultural and historical heritage and is an important global city. The city is also known for its nightlife, including the iconic nightlife districts of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of possibilities, but it's hard to predict exactly what it will look like. However, here are some possible trends that we could see in the coming years:
    
    1. Increased integration of AI into all aspects of our lives: AI will become even more integrated into our daily lives, from home automation to transportation to healthcare. It will help us make more informed decisions and streamline our work processes.
    
    2. AI will become more intelligent: We will see more sophisticated AI systems that can think, learn and adapt to new situations. This could lead to new forms of AI, such as "cognitive AI" or "machine learning," that can


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

     Alex

    ,

     and

     I

    'm

     a

     curious

     and

     adventurous

     person

     who

     loves

     exploring

     new

     places

     and

     learning

     new

     things

    .

     I

     love

     to

     travel

    ,

     read

    ,

     and

     explore

     new

     cultures

    .

     I

     have

     a

     natural

     curiosity

     and

     love

     to

     solve

     puzzles

     and

     find

     hidden

     meanings

     in

     everything

     I

     see

    .

     I

    'm

     a

     problem

     solver

    ,

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     solve

     challenges

    .

     I

    'm

     ready

     to

     join

     any

     adventure

     or

     challenge

     that

     comes

     my

     way

    !

     What

    's

     your

     name

    ?

     What

    's

     your

     profession

    ?

     What

    's

     your

     favorite

     hobby

    ?

     What

    's

     your

     longest

     adventure

    ?

     I

     hope

     to

     meet

     you

     soon

    !

     //

     Last

     seen

     in

     a

     photograph

     from

     the

     famous

     novel

     The

     Da

     Vinci

     Code

    .

     

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     encaps

    ulates

     the

     fundamental

     facts

     about

     Paris

    ,

     emphasizing

     its

     role

     as

     the

     nation

    's

     capital

     and

     the

     most

     important

     city

     in

     the

     country

    .

     It

     highlights

     the

     significance

     of

     the

     city

     in

     French

     culture

    ,

     politics

    ,

     and

     economy

    ,

     while

     also

     providing

     a

     concise

     overview

     of

     its

     importance

     to

     the

     nation

     as

     a

     whole

    .

     The

     statement

     is

     precise

    ,

     clear

    ,

     and

     informative

    ,

     providing

     readers

     with

     a

     basic

     understanding

     of

     Paris

    's

     place

     in

     French

     and

     global

     politics

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     combination

     of

     emerging

     technologies

     and

     the

     increasing

     complexity

     of

     problems

     that

     AI

     systems

     must

     solve

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

     Enhanced

     machine

     learning

    :

     As

     AI

     systems

     become

     more

     capable

     of

     learning

     from

     data

    ,

     they

     are

     likely

     to

     become

     even

     more

     sophisticated

    .

     This

     will

     require

     a

     continued

     focus

     on

     improving

     algorithms

    ,

     developing

     new

     approaches

    ,

     and

     exploring

     new

     machine

     learning

     techniques

    .
    


    2

    .

     AI

     ethics

     and

     safety

    :

     As

     AI

     systems

     become

     more

     integrated

     into

     everyday

     life

    ,

     there

     will

     be

     increasing

     demand

     for

     frameworks

     that

     ensure

     their

     safety

     and

     ethical

     conduct

    .

     This

     could

     involve

     new

     approaches

     to

     data

     privacy

    ,

     liability

    ,

     and

     accountability

    .
    


    3

    .

     Autonomous

     systems

    



```python
llm.shutdown()
```
