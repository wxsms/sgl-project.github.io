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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]


    2026-04-28 01:19:29,645 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 01:19:29] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=5632):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3328):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=3072):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=2816):  10%|█         | 6/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:09,  4.78it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:05<00:09,  4.78it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:05<00:09,  4.78it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:05<00:09,  4.78it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:05<00:09,  4.78it/s] Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:05<00:09,  4.78it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03,  9.31it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03,  9.31it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03,  9.31it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03,  9.31it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03,  9.31it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03,  9.31it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03,  9.31it/s]

    Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03,  9.31it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03,  9.31it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:05<00:03,  9.31it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 21.72it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 21.72it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 21.72it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 21.72it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 21.72it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 21.72it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 21.72it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 21.72it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 21.72it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 21.72it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 29.28it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 29.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.56it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=121.03 GB):   7%|▋         | 4/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.03 GB):   7%|▋         | 4/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.02 GB):   7%|▋         | 4/58 [00:00<00:02, 18.42it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=121.02 GB):  10%|█         | 6/58 [00:00<00:04, 12.30it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.01 GB):  10%|█         | 6/58 [00:00<00:04, 12.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.01 GB):  10%|█         | 6/58 [00:00<00:04, 12.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.01 GB):  10%|█         | 6/58 [00:00<00:04, 12.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.01 GB):  10%|█         | 6/58 [00:00<00:04, 12.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.01 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=121.00 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=121.00 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.19it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=120.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.99 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.99 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.98 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.98 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.98 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.97 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.95 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.59it/s]Capturing num tokens (num_tokens=960 avail_mem=120.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.59it/s] Capturing num tokens (num_tokens=896 avail_mem=120.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.59it/s]

    Capturing num tokens (num_tokens=832 avail_mem=120.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.59it/s]Capturing num tokens (num_tokens=768 avail_mem=120.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.59it/s]Capturing num tokens (num_tokens=768 avail_mem=120.96 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.47it/s]Capturing num tokens (num_tokens=704 avail_mem=120.96 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.47it/s]Capturing num tokens (num_tokens=640 avail_mem=120.95 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.47it/s]Capturing num tokens (num_tokens=576 avail_mem=120.95 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.47it/s]Capturing num tokens (num_tokens=512 avail_mem=120.94 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.47it/s]Capturing num tokens (num_tokens=480 avail_mem=120.95 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.47it/s]Capturing num tokens (num_tokens=480 avail_mem=120.95 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.26it/s]Capturing num tokens (num_tokens=448 avail_mem=120.95 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.26it/s]Capturing num tokens (num_tokens=416 avail_mem=120.95 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.26it/s]Capturing num tokens (num_tokens=384 avail_mem=120.95 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.26it/s]

    Capturing num tokens (num_tokens=352 avail_mem=120.94 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.26it/s]Capturing num tokens (num_tokens=320 avail_mem=120.94 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.26it/s]Capturing num tokens (num_tokens=320 avail_mem=120.94 GB):  60%|██████    | 35/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=288 avail_mem=120.93 GB):  60%|██████    | 35/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=256 avail_mem=120.93 GB):  60%|██████    | 35/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=240 avail_mem=120.93 GB):  60%|██████    | 35/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=224 avail_mem=120.92 GB):  60%|██████    | 35/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=208 avail_mem=120.92 GB):  60%|██████    | 35/58 [00:01<00:00, 39.85it/s]Capturing num tokens (num_tokens=208 avail_mem=120.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=192 avail_mem=120.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=176 avail_mem=120.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.31it/s]

    Capturing num tokens (num_tokens=160 avail_mem=120.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=144 avail_mem=120.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=128 avail_mem=120.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.31it/s]Capturing num tokens (num_tokens=128 avail_mem=120.91 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.65it/s]Capturing num tokens (num_tokens=112 avail_mem=120.91 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.65it/s]

    Capturing num tokens (num_tokens=96 avail_mem=120.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.65it/s] Capturing num tokens (num_tokens=80 avail_mem=120.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.65it/s]Capturing num tokens (num_tokens=64 avail_mem=120.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.65it/s]Capturing num tokens (num_tokens=48 avail_mem=120.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.65it/s]Capturing num tokens (num_tokens=48 avail_mem=120.89 GB):  86%|████████▌ | 50/58 [00:01<00:00, 30.41it/s]Capturing num tokens (num_tokens=32 avail_mem=120.89 GB):  86%|████████▌ | 50/58 [00:01<00:00, 30.41it/s]Capturing num tokens (num_tokens=28 avail_mem=120.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 30.41it/s]Capturing num tokens (num_tokens=24 avail_mem=120.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 30.41it/s]Capturing num tokens (num_tokens=20 avail_mem=120.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 30.41it/s]Capturing num tokens (num_tokens=16 avail_mem=120.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 30.41it/s]

    Capturing num tokens (num_tokens=16 avail_mem=120.88 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.64it/s]Capturing num tokens (num_tokens=12 avail_mem=120.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.64it/s]Capturing num tokens (num_tokens=8 avail_mem=120.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.64it/s] Capturing num tokens (num_tokens=4 avail_mem=120.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.64it/s]Capturing num tokens (num_tokens=4 avail_mem=120.87 GB): 100%|██████████| 58/58 [00:01<00:00, 31.98it/s]


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
    Generated text:  Isabelle and I'm a professional HR consultant and a language coach. I provide services in the UK and the USA, but I also have extensive experience in Europe, Asia and other regions. I specialize in providing my services through the Microsoft Office Live Meeting, but I am also available for clients in other platforms.
    
    I am currently based in London, but I have experience working in various countries. I am also proficient in French and German, and I have experience in working with clients in other languages.
    
    I have a bachelor's degree in English and a master's degree in Business and Management. I have experience in coaching and leadership, and I have
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the vice president. This is because
    
    A. The vice president has the most power in the White House.
    
    B. The vice president and the president are two different offices.
    
    C. There are no presidential offices.
    
    D. The vice president holds no power in the United States. To determine why the vice president is represented by the president, let's analyze each option step by step:
    
    A. The vice president has the most power in the White House.
    - This statement is not true. The vice president does not have the most power in the White House; the president does. The vice president is a subordinate position within the executive
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A: Paris
    B: London
    C: Berlin
    D: Moscow
    To determine the capital of France, we need to consider the information that France is a member of the European Union and is known for its influence and importance in the European region.
    
    The European Union is headquartered in Brussels, which is the capital of Belgium. However, Belgium is not part of France's territory. Therefore, the capital of France is not Brussels.
    
    The capital of France is not Berlin because it is not a member of the European Union.
    
    The capital of France is Paris, which is the capital of France and is located in the heart of the city
    ===============================
    Prompt: The future of AI is
    Generated text:  being shaped by the rapidly evolving landscape of technology and society. As we move forward, it is essential to consider the ethical implications of AI and how they will shape our future. Some of the most pressing issues include privacy, bias, and the impact of AI on employment.
    Privacy is a critical issue that is often overlooked in the development of AI. As AI becomes more advanced, it is likely to collect more data and information from individuals, which raises concerns about privacy and data security. One solution is to implement strong data protection policies and regulations to ensure that data is handled securely and ethically.
    Bias is another issue that is concerning. AI systems


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I enjoy [insert a short description of your hobbies or interests]. What's your favorite hobby or activity? I'm always looking for new experiences and adventures, so I enjoy [insert a short description of your favorite activity]. What's your favorite book or movie? I love [insert a short description of your favorite book or movie]. What's your favorite place to go? I love [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in Europe and the second-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also a major center for business, finance, and tourism, and is a popular tourist destination. The city is known for its cuisine, including French cuisine, and is home to many museums, theaters, and art galleries. Paris is a city of contrasts
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into many aspects of our lives, from smart home devices to self-driving cars. As AI continues to improve, we can expect to see even more integration into our daily routines.
    
    2. AI becoming more autonomous: As AI becomes more advanced, we can expect to see more autonomous vehicles on the roads, drones delivering packages, and other forms of automation that rely on AI.
    
    3. AI becoming more ethical and responsible: As AI
    


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
    Generated text:  [insert name of the character] and I'm [insert character's age]. I'm a [insert profession or occupation] who has been working in the [insert field of work] for [insert number of years] years. I love [insert one or two hobbies or interests that you know] and I have a very [insert one or two traits or skills] personality. I'm [insert some about how I would introduce myself without using the character's name, such as: "I'm a friendly and reliable friend who always has a smile on my face. " or "I'm a driven and ambitious person who is always eager
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To elaborate, Paris is the largest city in France, located in the northwestern suburbs of the country, and is considered a cosmopolitan metropolis with a rich cultural heritage. The city is known for its romantic atmosphere, historical sites, and its role as a cultural hub for Europe. 
    
    Paris is also renowned for its high standards of living and luxury, which reflect its status as a major financial center. Its proximity to neighboring cities and its role as a gateway to the city of Paris are notable features of its fame. The city has a large, diverse population, which has contributed to its multicultural character. 
    
    As the capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be driven by several trends and innovations that are expected to shape the direction of AI research, development, and deployment. Some of the key trends include:
    
    1. **Enhanced Model Performance**: Advances in machine learning algorithms and neural network architecture are expected to lead to more accurate and efficient AI models. This includes improvements in techniques like deep reinforcement learning, transfer learning, and model compression.
    
    2. **Neural Network Architecture**: The development of more advanced neural network architectures, such as transformers, graph neural networks, and recurrent neural networks, is expected to enhance AI performance. These architectures can model complex relationships between inputs, making them capable of


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

    ],

     and

     I

    'm

     a

     [

    职业

    ]

     who

     loves

     [

    职业

    的

    爱好

    ]

     and

     enjoys

     [

    职业

    的乐趣

    或

    挑战

    ]

     for

     fun

     and

     relaxation

    .

     I

    'm

     patient

     and

     have

     a

     natural

     inclination

     to

     be

     calm

    ,

     and

     I

    'm

     always

     ready

     to

     learn

     and

     grow

     with

     each

     new

     opportunity

    .

     I

    'm

     a

     team

     player

     and

     am

     always

     willing

     to

     help

     others

    ,

     so

     if

     you

     need

     anything

    ,

     I

    'm

     always

     here

     to

     assist

    .

     My

     goal

     is

     to

     make

     a

     positive

     impact

     on

     the

     world

     and

     inspire

     others

     to

     do

     the

     same

    .

     So

    ,

     please

     introduce

     yourself

     to

     me

     and

     let

    's

     see

     if

     we

     can

     become

     a

     great

     team

    .

     [

    Name

    ]

     [

    Introduction

     and

     opening

     statement

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     with

     a

     rich

     history

     dating

     back

     over

     

    2

    ,

    0

    0

    0

     years

    .

     It

     is

     known

     for

     its

     medieval

     architecture

    ,

     grand

     bou

    lev

    ards

    ,

     and

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     home

     to

     the

     world

    -f

    amous

     Paris

     Opera

     and

     the

     Bast

    ille

    ,

     a

     symbol

     of

     French

     power

     and

     freedom

    .

     The

     city

     is

     a

     vibrant

     hub

     of

     culture

     and

     commerce

    ,

     with

     a

     vibrant

     arts

     scene

     and

     numerous

     festivals

     and

     events

     throughout

     the

     year

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     its

     bustling

     commercial

     districts

     and

     picturesque

     countryside

    ,

     making

     it

     a

     popular

     tourist

     destination

     for

     millions

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     quite

     exciting

     and

     diverse

    ,

     with

     several

     trends

     poised

     to

     shape

     how

     we

     use

     and

     benefit

     from

     AI

     technology

    .

     Here

     are

     some

     of

     the

     most

     likely

     developments

    :
    


    1

    .

     Increased

     focus

     on

     ethics

     and

     privacy

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

     increasing

     scrutiny

     of

     how

     it

     is

     being

     used

     and

     regulated

    .

     Governments

     and

     organizations

     will

     need

     to

     prioritize

     the

     ethical

     and

     privacy

     concerns

     of

     AI

     development

     and

     deployment

    ,

     ensuring

     that

     it

     does

     not

     infr

    inge

     on

     human

     rights

     or

     create

     harmful

     biases

    .
    


    2

    .

     AI

     development

     will

     continue

     to

     improve

    ,

     but

     it

     will

     also

     face

     new

     challenges

    :

     AI

     is

     becoming

     more

     complex

     and

     capable

    ,

     but

     it

     will

     also

     face

     new

     challenges

     such

    



```python
llm.shutdown()
```
