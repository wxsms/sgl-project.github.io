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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.25it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.24it/s]


    2026-04-06 06:32:48,531 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 06:32:48] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:18,  2.43s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:18,  2.43s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:18,  2.43s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:18,  2.43s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:26,  2.05it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:26,  2.05it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:26,  2.05it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:26,  2.05it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:26,  2.05it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:26,  2.05it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:26,  2.05it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.24it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.24it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.24it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.24it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.24it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.24it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.24it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.24it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.24it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.24it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.89it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.89it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.89it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.89it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.89it/s]

    Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.89it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.89it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:02<00:01, 19.22it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:02<00:01, 19.22it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:02<00:01, 19.22it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:02<00:01, 19.22it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:02<00:01, 19.22it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:02<00:01, 19.22it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:02<00:01, 19.22it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:02<00:01, 19.22it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:02<00:00, 26.33it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:02<00:00, 26.33it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:02<00:00, 26.33it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:02<00:00, 26.33it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:00, 26.33it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:00, 26.33it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:00, 26.33it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:00, 26.33it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 33.64it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 33.64it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 33.64it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 33.64it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 33.64it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 33.64it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 33.64it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 33.64it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 33.64it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 41.32it/s]

    Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 41.32it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 51.40it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 51.40it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 51.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=49.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.74 GB):   3%|▎         | 2/58 [00:00<00:02, 19.92it/s]Capturing num tokens (num_tokens=7168 avail_mem=49.73 GB):   3%|▎         | 2/58 [00:00<00:02, 19.92it/s]Capturing num tokens (num_tokens=6656 avail_mem=49.73 GB):   3%|▎         | 2/58 [00:00<00:02, 19.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=49.73 GB):   3%|▎         | 2/58 [00:00<00:02, 19.92it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=49.73 GB):   9%|▊         | 5/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=49.73 GB):   9%|▊         | 5/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=49.73 GB):   9%|▊         | 5/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=49.73 GB):   9%|▊         | 5/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=49.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=49.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.94it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=53.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.58 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.58 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.58 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.58 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.57 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.57 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.96it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=53.56 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.54 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=960 avail_mem=53.56 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.96it/s] Capturing num tokens (num_tokens=960 avail_mem=53.56 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=896 avail_mem=53.56 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=832 avail_mem=53.55 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=768 avail_mem=53.55 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=704 avail_mem=53.55 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=640 avail_mem=53.54 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=576 avail_mem=53.54 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=576 avail_mem=53.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=512 avail_mem=53.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=480 avail_mem=53.55 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.47it/s]

    Capturing num tokens (num_tokens=448 avail_mem=53.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=416 avail_mem=53.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=384 avail_mem=53.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=352 avail_mem=53.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.47it/s]Capturing num tokens (num_tokens=352 avail_mem=53.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=320 avail_mem=53.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=288 avail_mem=53.53 GB):  59%|█████▊    | 34/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=256 avail_mem=53.52 GB):  59%|█████▊    | 34/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=240 avail_mem=53.52 GB):  59%|█████▊    | 34/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=224 avail_mem=53.52 GB):  59%|█████▊    | 34/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=208 avail_mem=53.51 GB):  59%|█████▊    | 34/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=208 avail_mem=53.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.35it/s]Capturing num tokens (num_tokens=192 avail_mem=53.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.35it/s]

    Capturing num tokens (num_tokens=176 avail_mem=53.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.35it/s]Capturing num tokens (num_tokens=160 avail_mem=53.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.35it/s]Capturing num tokens (num_tokens=144 avail_mem=53.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.35it/s]Capturing num tokens (num_tokens=128 avail_mem=53.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.35it/s]Capturing num tokens (num_tokens=112 avail_mem=53.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 48.35it/s]Capturing num tokens (num_tokens=112 avail_mem=53.50 GB):  79%|███████▉  | 46/58 [00:01<00:00, 50.18it/s]Capturing num tokens (num_tokens=96 avail_mem=53.50 GB):  79%|███████▉  | 46/58 [00:01<00:00, 50.18it/s] Capturing num tokens (num_tokens=80 avail_mem=53.49 GB):  79%|███████▉  | 46/58 [00:01<00:00, 50.18it/s]Capturing num tokens (num_tokens=64 avail_mem=53.49 GB):  79%|███████▉  | 46/58 [00:01<00:00, 50.18it/s]Capturing num tokens (num_tokens=48 avail_mem=53.49 GB):  79%|███████▉  | 46/58 [00:01<00:00, 50.18it/s]Capturing num tokens (num_tokens=32 avail_mem=53.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 50.18it/s]Capturing num tokens (num_tokens=28 avail_mem=53.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 50.18it/s]

    Capturing num tokens (num_tokens=28 avail_mem=53.48 GB):  90%|████████▉ | 52/58 [00:01<00:00, 51.07it/s]Capturing num tokens (num_tokens=24 avail_mem=53.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 51.07it/s]Capturing num tokens (num_tokens=20 avail_mem=53.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 51.07it/s]Capturing num tokens (num_tokens=16 avail_mem=53.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 51.07it/s]Capturing num tokens (num_tokens=12 avail_mem=53.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 51.07it/s]Capturing num tokens (num_tokens=8 avail_mem=53.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 51.07it/s] Capturing num tokens (num_tokens=4 avail_mem=53.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 51.07it/s]Capturing num tokens (num_tokens=4 avail_mem=53.46 GB): 100%|██████████| 58/58 [00:01<00:00, 52.38it/s]Capturing num tokens (num_tokens=4 avail_mem=53.46 GB): 100%|██████████| 58/58 [00:01<00:00, 40.71it/s]


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
    Generated text:  Yuwen Hu (Orchid). I am a Master of the Skill of the Sacred Sun, which is located at the middle of the south of the Earth. Currently, I am performing a ceremony that has been long overdue for a long time.
    I am currently preparing for the Summer Solstice, the moment when the Sun is at its highest point. I am still a long way from being ready for this ceremony. I have been practicing every day for the past 12 years, but it hasn't been a success. Perhaps the cause of this is that my meditation skills are not yet proficient, and my practice is not aligned with
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a way to prevent the country from becoming too reliant on oil. What are some possible solutions he could consider? Answer according to: Theodore Roosevelt: An American President – A Life. Theodore Roosevelt was a member of the Republican Party and served as the 29th president of the United States from 1901 to 1909. He is known for his leadership of the conservation movement. During his presidency, he signed the first federal land trust act in the history of the United States. Roosevelt was the first president to visit the front lines in World War I and was involved in a number of legislative and executive
    ===============================
    Prompt: The capital of France is
    Generated text:  the
    
    A) Paris
    B) Lille
    C) Lyon
    D) Nancy
    
    The correct answer is A) Paris. Paris is the capital of France, which is located in the center of France. It is the largest city in France by population and is known for its rich history, culture, and beautiful architecture. 
    
    The other options are not capitals of France:
    - Lille is the capital of France and is located in the southwestern part of the country.
    - Lyon is the capital of France and is located in the southeast of the country.
    - Nancy is the capital of France and is located in the northwestern part of
    ===============================
    Prompt: The future of AI is
    Generated text:  not the future of the industry. It is the future of human beings. In my book, The Great Courses: AI, AI is changing the way we do business, learn, and live, transforming our world. But at its core, AI is about how we do business and learn. In this course, I go beyond the surface level to explore how AI is altering the way we do business and what that means for the way we work with AI. The new AI that has been developing over the last decade is transforming the ways we work with AI, the data we collect and work with, and the way we interact with AI. We


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is the largest city in France by population. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its cuisine, fashion, and art scene. It is a popular tourist destination and a major economic center in France. The city is home to many world-renowned museums, theaters, and other cultural institutions. Paris is a vibrant and dynamic city with a rich history and a strong sense of community. Its status as the capital of France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This could lead to more sophisticated forms of AI that can learn and adapt to new situations, rather than simply following pre-programmed instructions.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations. This could lead to more transparent and accountable AI systems
    


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
    Generated text:  [Name] and I am a [gender] [name] [age]. I am a [occupation] [title], and I am passionate about [what I enjoy doing]. If you're ever in need of a [service or project], I'm your go-to person for [why I'm the best]. I also have a [attraction or personality] that makes me stand out among my peers. 
    
    I'm a [time period] [nationality] native [city or country] living in [current city or country]. I have a [accomplishment or skill] that I'm proud of, and I'm always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city and also the largest metropolitan area in the country. Paris is a city with rich history and culture, and it is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre Dame Cathedral. It is also the birthplace of many famous figures, including Napoleon Bonaparte, Louis XVI, and Marie Antoinette. Paris is also known for its fashion industry, with many famous Parisian haute couture designers. It is a popular tourist destination and home to the world-renowned Eiffel Tower, the Louvre Museum, and other museums. Paris is the cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a number of trends that will shape how it is used and developed. Some of the most significant trends include:
    
    1. Deep learning: With the development of deep learning techniques, artificial intelligence will become more powerful and capable of solving increasingly complex problems. This will lead to applications in areas such as image and speech recognition, natural language processing, and predictive analytics.
    
    2. Automation and robotics: As technology continues to advance, it is likely that robots and automation will become more integrated into our daily lives, including in manufacturing, transportation, and logistics. This will require significant changes in the way we work and live.
    
    3.


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

     [

    Your

     Title

    /

    Role

    ].

     I

     enjoy

     [

    Your

     Inter

    ests

    /

    Activities

    /

    Ch

    allenges

    ].

     I

    'm

     passionate

     about

     [

    Your

     Career

    /

    Projects

    /

    Goals

    ].

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     as

     a

     [

    Your

     Field

    /

    Subject

    /

    Role

    ].

     I

     thrive

     on

     learning

     and

     exploring

     new

     things

    .

     I

     am

     a

     [

    Your

     Character

     trait

    ].

     And

     last

     but

     not

     least

    ,

     I

    'm

     [

    Your

     Character

     Name

    ].

     I

    'm

     confident

    ,

     enthusiastic

    ,

     and

     always

     ready

     to

     tackle

     whatever

     comes

     my

     way

    .

     I

     love

     trying

     new

     things

     and

     never

     give

     up

     easily

    .

     I

    'm

     always

     up

     for

     a

     challenge

     and

     eager

     to

     push

     myself

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     vie

     de

     l

    '

    homme

    ".

     It

     is

     a

     cultural

     and

     economic

     center

     of

     the

     country

     and

     is

     known

     for

     its

     iconic

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

     E

    iff

    el

     Tower

    .

     The

     city

     is

     also

     home

     to

     many

     famous

     universities

     and

     museums

    ,

     and

     is

     a

     major

     transportation

     hub

     for

     Europe

    .

     Paris

     is

     a

     vibrant

     and

     diverse

     city

     with

     a

     rich

     history

     and

     a

     strong

     cultural

     heritage

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     economic

     hub

     in

     the

     country

    .

     The

     city

     is

     also

     known

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     wine

    ,

     making

     it

     a

     unique

     and

     exciting

     place

     to

     visit

    .

     As

     a

     result

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     depends

     on

     a

     variety

     of

     factors

    ,

     including

     the

     development

     of

     new

     technologies

     and

     the

     policies

     and

     regulations

     that

     govern

     them

    .

     However

    ,

     there

     are

     some

     potential

     trends

     that

     may

     be

     expected

     in

     the

     AI

     industry

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     emphasis

     on

     ethical

     considerations

    :

     As

     the

     AI

     industry

     continues

     to

     grow

     and

     become

     more

     integrated

     into

     various

     aspects

     of

     life

    ,

     there

     will

     be

     increasing

     pressure

     to

     ensure

     that

     AI

     systems

     are

     fair

     and

     equitable

    .

     This

     may

     lead

     to

     more

     emphasis

     on

     ethical

     and

     moral

     considerations

     when

     designing

     and

     testing

     AI

     systems

    .
    


    2

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

     increasingly

     being

     used

     in

     healthcare

     to

     improve

     patient

     outcomes

     and

     reduce

     costs

    .

     As

    



```python
llm.shutdown()
```
