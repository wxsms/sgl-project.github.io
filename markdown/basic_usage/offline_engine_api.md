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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.88it/s]


    2026-05-16 18:24:24,565 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 18:24:24] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:52,  4.08s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.27it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.27it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.27it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.27it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.27it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.27it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:05,  7.05it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:05,  7.05it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:05,  7.05it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:05,  7.05it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:05,  7.05it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:05,  7.05it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:04<00:05,  7.05it/s] 

    Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:04<00:05,  7.05it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.01it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.01it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.01it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.01it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.01it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.01it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 15.75it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 15.75it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 15.75it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 15.75it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 15.75it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 15.75it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 15.75it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 15.75it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 15.75it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s]

    Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:04<00:00, 33.79it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:04<00:00, 46.59it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:04<00:00, 46.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 42.79it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.79it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.79it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.79it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.79it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 42.79it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=288 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=224 avail_mem=76.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=224 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.70it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.70it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=160 avail_mem=76.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.70it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.38it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=64 avail_mem=76.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=64 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.36it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.36it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.36it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.36it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.36it/s]

    Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.36it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.68it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.68it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.68it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.68it/s] Capturing num tokens (num_tokens=4 avail_mem=76.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.68it/s]Capturing num tokens (num_tokens=4 avail_mem=76.60 GB): 100%|██████████| 58/58 [00:01<00:00, 41.63it/s]


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
    Generated text:  Marco. As you can see, I am going to be a regular guy and not a figure skater. I will be the subject of this story, and I will be the one who is talking about myself. I'm going to do a lot of questions about myself. I'm going to be real. But I'm going to be honest and not pretend. I want you to tell me what you know about yourself. What are your hobbies? Where do you go to the gym? What are you interested in doing in your free time? Is your favorite color? What's your favorite book? What's your favorite movie? What's
    ===============================
    Prompt: The president of the United States is
    Generated text:  a holder of a ____.
    A. office
    B. position
    C. title
    D. title
    Answer: A
    
    In a psychological experiment, the subjects were asked to complete a task in as many minutes as possible, and their performance was observed by a referee. The referee noted whether the subjects met the performance standards, and the outcomes were categorized as 'good' or 'bad'. Based on this experimental design, what is the primary variable being studied? 
    A. Reaction Time
    B. Subject's Performance
    C. Time Limit
    D. External Verbal Scores
    Answer: B
    
    The genetic material of a human being
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located in the southwest of the country. It was a capital city of the Kingdom of France from 1450 to 1790. Its population was 1,870,000 in 2017. Its coordinates are 48°58′N latitude and 2°20′E longitude. It is located in the west of the North Sea. The city is also the capital of the department of Paris. It is the most populous of the 500 cities in France. It has a population of 6,728,5
    ===============================
    Prompt: The future of AI is
    Generated text:  in the air, but the existing AI is not ready for the climate change crisis. If we are to truly harness the power of AI, we must first understand its limitations and limitations.
    The main reason for the current AI crisis is that AI has a limit to its abilities. The cloud-based AI model has been limited by the data it has access to and the size of its storage capacity. The typical cloud model has a maximum capacity of 1 terabyte, which is insufficient to store the data required to accurately simulate the climate crisis.
    That's why the current AI is not ready for the climate change crisis, and it will not be ready


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


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its rich history, art, and culture, and is home to many world-renowned museums, theaters, and festivals. Paris is a vibrant and dynamic city with a rich cultural scene, and is a popular tourist destination for visitors from around the world. The city is also home to many important political and economic institutions, including the French Parliament, the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced decision-making. This could lead to more personalized and context-aware AI that can better understand and respond to human emotions and behaviors.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more integrated with human intelligence, it is likely to be used in even more advanced ways, such as developing more accurate medical diagnoses and personalized treatment plans.
    
    3. Increased use of AI in finance: AI is already
    


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
    Generated text:  Emily. I love to travel and explore new places. I'm also a big fan of hiking and photography, and I have a passion for sustainability and environmental conservation. I'm always trying to learn more about the world around me and try to appreciate the beauty of nature. What's your favorite hobby or activity to do? I enjoy spending time with my family and friends, reading books and watching movies, and taking walks in nature. How do you stay healthy? I try to eat a healthy diet and exercise regularly, and I often go to the gym. What do you like to do when you're not working? I enjoy spending time with
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, stunning architecture, and vibrant culture. As the country's largest city, Paris is home to numerous famous landmarks such as the Eiffel Tower and the Louvre Museum. The city is also famous for its cuisine, including the famous Parisian poutine and croissants. Paris is also the home of the European Parliament and the city's annual gastronomic week, where local food vendors prepare dishes in competition. Despite being a massive city with a population of more than 2. 5 million, Paris remains a unique and charming city that has captured the attention of tourists and locals alike.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be one of continued evolution and adaptation, with various trends set to shape its direction and impact on society. Here are some potential trends to expect:
    
    1. Increased automation: One of the most significant trends in AI is the continued development of automation, which will likely become more pervasive in various sectors. This could include the automation of tasks such as data entry, customer service, and routine maintenance, as well as the use of AI-powered machines to perform routine tasks that can be performed by humans.
    
    2. Integration of AI with other technologies: AI is likely to become even more integrated with other technologies, such as sensors, data analytics,


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

     an

     AI

     assistant

     designed

     to

     help

     people

     find

     the

     information

     they

     need

    .

     I

     don

    't

     have

     emotions

     or

     personal

     experiences

    ,

     but

     I

    'm

     always

     here

     to

     help

     and

     answer

     any

     questions

     you

     may

     have

    .

     How

     can

     I

     assist

     you

     today

    ?

     [

    Type

     your

     response

     here

    ]

     [

    Name

    ]

     is

     a

     chat

    bot

     designed

     to

     assist

     users

     in

     finding

     information

     and

     answer

     their

     questions

    .

     I

     don

    't

     have

     emotions

     or

     personal

     experiences

    ,

     but

     I

    'm

     always

     here

     to

     help

     and

     answer

     any

     questions

     you

     may

     have

    .

     How

     can

     I

     assist

     you

     today

    ?

     [

    Type

     your

     response

     here

    ]

     [

    Name

    ]

     is

     a

     virtual

     assistant

     designed

     to

     assist

     users

     in

     finding

     information

     and

     answer

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     as

     the

     "

    City

     of

     a

     Thousand

     Squ

    ares

    "

     due

     to

     its

     extensive

     network

     of

     can

    als

     and

     cob

    ble

    stone

     streets

    .


    Paris

     is

     home

     to

     over

     

    5

    0

    0

    ,

     

    0

    0

    0

     people

    ,

     making

     it

     one

     of

     the

     largest

     cities

     in

     the

     world

     by

     population

    .

     The

     city

     is

     divided

     into

     

    6

    0

     districts

    ,

     each

     with

     its

     own

     distinct

     character

     and

     architecture

    .

     The

     E

    iff

    el

     Tower

    ,

     Arc

     de

     Tri

    omp

    he

    ,

     and

     Notre

    -D

    ame

     Cathedral

     are

     just

     a

     few

     of

     the

     most

     famous

     landmarks

    ,

     while

     the

     Lou

    vre

     Museum

     and

     the

     Sac

    ré

    -C

    œur

     Basil

    ica

     are

     also

     must

    -

    visit

     sites

    .

     Paris

     is

     also

     known

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     proliferation

     of

     smart

     technologies

     and

     applications

    .

     In

     the

     coming

     years

    ,

     we

     are

     likely

     to

     see

     more

     advanced

     AI

     that

     is

     capable

     of

     more

     complex

     tasks

     and

     tasks

     that

     are

     beyond

     the

     capabilities

     of

     current

     AI

     systems

    .
    


    One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     the

     increasing

     use

     of

     deep

     learning

    ,

     which

     is

     a

     subset

     of

     machine

     learning

     that

     involves

     training

     neural

     networks

     to

     make

     predictions

     and

     classifications

    .

     Deep

     learning

     is

     expected

     to

     become

     more

     prevalent

     as

     more

     data

     is

     collected

     and

     the

     complexity

     of

     tasks

     increases

    .
    


    Another

     trend

     in

     AI

     is

     the

     development

     of

     more

     advanced

     natural

     language

     processing

     (

    N

    LP

    )

     capabilities

    .

     N

    LP

     is

     the

     ability

     of

     AI

     systems

     to

     understand

    



```python
llm.shutdown()
```
