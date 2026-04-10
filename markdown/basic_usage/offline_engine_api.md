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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.43it/s]


    2026-04-10 14:28:38,703 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 14:28:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:03, 12.97it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:03, 12.97it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.97it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.97it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.97it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.97it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.97it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.97it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.97it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 27.68it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 27.68it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 27.68it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 27.68it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 27.68it/s]

    Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 27.68it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 27.68it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 27.68it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 34.17it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 34.17it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 34.17it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 34.17it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 34.17it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 34.17it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 34.17it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 34.17it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 40.32it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 40.32it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 40.32it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 40.32it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 40.32it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 40.32it/s]

    Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 40.32it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 40.32it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 40.32it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 40.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=71.05 GB):   2%|▏         | 1/58 [00:00<00:43,  1.30it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.02 GB):   2%|▏         | 1/58 [00:00<00:43,  1.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.02 GB):   2%|▏         | 1/58 [00:00<00:43,  1.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.02 GB):   5%|▌         | 3/58 [00:00<00:13,  3.98it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.01 GB):   5%|▌         | 3/58 [00:00<00:13,  3.98it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   5%|▌         | 3/58 [00:00<00:13,  3.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:01<00:08,  6.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:01<00:08,  6.46it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:01<00:08,  6.46it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):  12%|█▏        | 7/58 [00:01<00:05,  8.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):  12%|█▏        | 7/58 [00:01<00:05,  8.91it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):  12%|█▏        | 7/58 [00:01<00:05,  8.91it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  12%|█▏        | 7/58 [00:01<00:05,  8.91it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:01<00:03, 12.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:01<00:03, 12.54it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:01<00:03, 12.54it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  17%|█▋        | 10/58 [00:01<00:03, 12.54it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.50it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=70.99 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.99 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.99 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.41it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=70.98 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.98 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.82it/s]Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.82it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.82it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.82it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.80it/s]Capturing num tokens (num_tokens=768 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.80it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.80it/s]

    Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.80it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.80it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.23it/s]Capturing num tokens (num_tokens=512 avail_mem=70.94 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.23it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.23it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:01<00:01, 26.23it/s]Capturing num tokens (num_tokens=416 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:02<00:01, 26.23it/s]Capturing num tokens (num_tokens=384 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:02<00:01, 26.23it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  48%|████▊     | 28/58 [00:02<00:01, 26.23it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:02<00:00, 33.16it/s]Capturing num tokens (num_tokens=320 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:02<00:00, 33.16it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:02<00:00, 33.16it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:02<00:00, 33.16it/s]

    Capturing num tokens (num_tokens=240 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:02<00:00, 33.16it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:02<00:00, 33.16it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:02<00:00, 33.16it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  69%|██████▉   | 40/58 [00:02<00:00, 38.69it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  69%|██████▉   | 40/58 [00:02<00:00, 38.69it/s]Capturing num tokens (num_tokens=176 avail_mem=70.92 GB):  69%|██████▉   | 40/58 [00:02<00:00, 38.69it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:02<00:00, 38.69it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:02<00:00, 38.69it/s]Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:02<00:00, 38.69it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:02<00:00, 38.69it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  79%|███████▉  | 46/58 [00:02<00:00, 42.29it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:02<00:00, 42.29it/s] Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:02<00:00, 42.29it/s]

    Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:02<00:00, 42.29it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:02<00:00, 42.29it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:02<00:00, 42.29it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  79%|███████▉  | 46/58 [00:02<00:00, 42.29it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:02<00:00, 44.77it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:02<00:00, 44.77it/s]Capturing num tokens (num_tokens=20 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:02<00:00, 44.77it/s]Capturing num tokens (num_tokens=16 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:02<00:00, 44.77it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  90%|████████▉ | 52/58 [00:02<00:00, 44.77it/s]Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  90%|████████▉ | 52/58 [00:02<00:00, 44.77it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  90%|████████▉ | 52/58 [00:02<00:00, 44.77it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:02<00:00, 47.12it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:02<00:00, 22.92it/s]


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
    Generated text:  Alicia, I am a 28 year old person who really likes writing. I also like reading and participating in writing groups. I am writing the story "The Rise of the Immortals" which is about a group of characters who battle the other immortals. 
    
    I am searching for a software that could help me to organize my writing ideas. I have a lot of writing projects, so I was wondering if there is a software that could help me to write out my stories in a certain format. For example, I would like to have it in a specific order or format. I can only use this software on my computer
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is like a leader of the country. He has a lot of important duties to do every day. 
    
    The president of the United States has two important jobs:  
    1. The first job is to try to keep the country safe.  
    2. The second job is to work for the people and help them by sending letters to them asking for help. He is like a messenger for the people.
    
    He is also very important because he is a leader of the country. He works hard to make sure that the people of the United States are happy, healthy, and comfortable. He is the president of the United
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the ____ city in France.
    A. largest
    B. second largest
    C. smallest
    D. third smallest
    
    To determine the correct answer, let's first understand what the question is asking. The question is about the capital of France and asks which of the following is the correct adjective to use before "city" to describe it.
    
    1. **Largest**: This would be used if the capital were a very large city, but Paris is not the largest city in France.
    2. **Second largest**: This would be used if the capital were a city that was not the largest in France, but Paris was
    ===============================
    Prompt: The future of AI is
    Generated text:  very promising and it is very important for us to understand the big potential of AI. However, in order for us to fully utilize AI in our lives, we must understand its limitations. For example, in the field of healthcare, AI has made significant strides in medical diagnosis and treatment, but it has also raised concerns about the ethical implications of AI in our daily lives. One of the main challenges with AI is ensuring that its development is safe and that it is not used in ways that harm people.
    To address this challenge, it is essential to develop AI systems that are designed to be ethical and transparent. This means that they should be designed


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and being home to many famous artists, writers, and musicians. Paris is a popular tourist destination, with millions of visitors annually, making it one of the most visited cities in the world. The city is also known for its diverse cuisine, including French cuisine, as well as international cuisine. Paris is a city of contrasts, with its historic architecture and modern fashion, and is a must-visit destination
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. Greater integration with human intelligence: AI will continue to become more integrated with human intelligence, allowing for more complex and nuanced interactions between humans and machines.
    
    3. Increased use of AI in healthcare: AI will be used to improve the accuracy and efficiency of medical diagnosis and
    


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
    Generated text:  [Name], and I'm [Age]. I am [Occupation/Subject] and have always been passionate about [what I like to do]. I have always wanted to share my knowledge and experiences with others, and I am eager to share my ideas and insights with you. I am always open to learning and growing in all aspects of life, and I am always looking for ways to make a positive impact on the world. I am always grateful for the opportunity to be around others and to help them along the way. Thanks for taking the time to meet me. Let me know if you have any questions. [Name] [Phone
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the most populous city in France and is the capital of France. It is located in the western part of the country, on the Île de la Cité, and is surrounded by the City of Paris and the Seine River. The city is home to the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, among other landmarks. Paris is known for its rich culture, culinary heritage, and historical significance, making it a popular tourist destination worldwide. Its proximity to the Seine River, the romantic city of Versailles, and its role as a major financial center make it a major hub
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  increasingly projected to be influenced by several key trends, including:
    
    1. Increased automation and efficiency: As AI continues to become more advanced and capable, it is projected to automate a variety of tasks and processes, including customer service, manufacturing, and production. This could lead to significant improvements in efficiency, cost savings, and overall productivity.
    
    2. Development of machine learning and deep learning: These are two key areas of AI that are expected to continue to advance. Machine learning involves the development of algorithms that can learn and improve over time, while deep learning involves the use of neural networks that can process complex data.
    
    3. Greater focus on ethics and


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

     am

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

     the

     field

     of

     [

    industry

    ].

     I

     have

     a

     keen

     interest

     in

     [

    specific

     skill

     or

     area

     of

     expertise

    ]

     and

     I

     love

     helping

     others

     achieve

     their

     goals

    .

     I

     am

     committed

     to

     [

    specific

     goal

     or

     mission

    ]

     and

     I

     am

     always

     looking

     to

     learn

     new

     things

     and

     stay

     up

    -to

    -date

     with

     the

     latest

     trends

     and

     techniques

    .

     I

     believe

     that

     [

    reason

     for

     being

     a

     good

     fit

     with

     the

     character

    ],

     and

     I

     am

     confident

     in

     my

     ability

     to

     contribute

     to

     [

    the

     character

    's

     organization

     or

     cause

    ]

     and

     to

     provide

     exceptional

     service

     and

     results

    .

     I

     am

     eager

     to

     contribute

     to

     a

     positive

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Fill

     in

     the

     blanks

     with

     the

     correct

     option

    .
    


    **

    A

    )

     has

     a

     small

     population

      


    B

    )

     is

     the

     most

     populous

     city

     in

     Europe

      


    C

    )

     has

     a

     population

     of

     more

     than

     

    1

     million

     people

      


    D

    )

     has

     the

     most

     expensive

     real

     estate

     prices

     in

     the

     world

    **

     
    


    Select

     the

     correct

     answer

    :
    


    A

    )

      


    B

    )

      


    C

    )

      


    D

    )

      


    E

    )
    


    To

     determine

     the

     correct

     answer

    ,

     we

     need

     to

     analyze

     the

     statement

     about

     Paris

    ,

     which

     is

     the

     capital

     of

     France

    .

     Let

    's

     examine

     each

     option

    :
    


    A

    )

     "

    has

     a

     small

     population

    "

     -

     This

     is

     incorrect

     because

     Paris

     is

     a

     very

     large

     city

    ,

     and

     its

     population

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     likely

     to

     continue

     to

     be

     a

     rapidly

     evolving

     field

    .

     Here

     are

     some

     potential

     trends

     that

     are

     likely

     to

     shape

     the

     AI

     landscape

     in

     the

     years

     to

     come

    :
    


    1

    .

     Increased

     Automation

    :

     AI

     is

     likely

     to

     become

     more

     prevalent

     in

     routine

    ,

     repetitive

     tasks

    ,

     such

     as

     data

     entry

    ,

     voice

     recognition

    ,

     and

     natural

     language

     processing

    .

     As

     AI

     becomes

     more

     adept

     at

     performing

     these

     tasks

    ,

     it

     is

     likely

     to

     automate

     other

     jobs

     and

     free

     up

     human

     resources

     to

     focus

     on

     more

     complex

     tasks

    .
    


    2

    .

     Enhanced

     Intelligence

    :

     The

     AI

     industry

     is

     moving

     towards

     developing

     algorithms

     that

     can

     become

     more

     intelligent

     over

     time

    .

     As

     we

     learn

     more

     about

     AI

    ,

     it

     is

     likely

     to

     become

     better

     at

     recognizing

     patterns

    



```python
llm.shutdown()
```
