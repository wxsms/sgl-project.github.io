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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.12it/s]


    2026-05-16 14:13:43,487 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 14:13:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.24it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.24it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.24it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.24it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.24it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.24it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.24it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  8.24it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:04,  8.24it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:04<00:04,  8.24it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 14.43it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 14.43it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 14.43it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 14.43it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 14.43it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 14.43it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 14.43it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 14.43it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 14.43it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:02, 14.43it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 21.83it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 21.83it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 21.83it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 21.83it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 21.83it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 21.83it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 21.83it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 21.83it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 21.83it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 21.83it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:04<00:00, 21.83it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 31.17it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 31.17it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 31.17it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 31.17it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 31.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 42.73it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.83it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.83it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.83it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.83it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.29it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 35.29it/s] Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.60it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.60it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.60it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.60it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.60it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.60it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.30it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.17it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.17it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.17it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.17it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.17it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 34.60it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  67%|██████▋   | 39/58 [00:01<00:00, 26.36it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 26.36it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 26.36it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 26.36it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 25.67it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 25.67it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 25.67it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 25.67it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 25.67it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.25it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.25it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.25it/s]Capturing num tokens (num_tokens=64 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.25it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.25it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.09it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.09it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.09it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.09it/s]Capturing num tokens (num_tokens=20 avail_mem=74.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.09it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 30.17it/s]Capturing num tokens (num_tokens=16 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 30.17it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 30.17it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 30.17it/s] Capturing num tokens (num_tokens=4 avail_mem=74.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 30.17it/s]Capturing num tokens (num_tokens=4 avail_mem=74.24 GB): 100%|██████████| 58/58 [00:01<00:00, 31.41it/s]Capturing num tokens (num_tokens=4 avail_mem=74.24 GB): 100%|██████████| 58/58 [00:01<00:00, 30.63it/s]


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
    Generated text:  Semyon and I have just graduated from the University of Applied Sciences, Leningrad. I am an educator and an entrepreneur who has always had a passion for teaching. I am now pursuing my first venture as an entrepreneur. I recently started my own business "Emotionv", which offers a comprehensive coaching program for beginners in the areas of music, language learning, and business. My aim is to offer a holistic approach that addresses the emotional, cognitive, and physical needs of the participants. 
    
    Please provide a list of the unique features and benefits of my coaching program "Emotionv" for beginners. The program is designed to be
    ===============================
    Prompt: The president of the United States is
    Generated text:  from which country?
    The president of the United States is from the United States of America, which is a federal republic. The current president is Joe Biden, who has been in office since 2021. He was born on December 22, 1961, and has been in office during a period of significant political and economic changes in the country. The country is known for its democracy, cultural diversity, and technological advancements.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the capital of the United Kingdom is London, the capital of the United States is Washington D. C. and the capital of the United States of America is Washington, D. C. Does this mean that the capital of the United States of America is the capital of France?
    
    To determine whether the capital of the United States of America is the capital of France, we need to break down the information provided and analyze it step by step.
    
    1. Identify the capital of France.
       The capital of France is Paris.
    
    2. Identify the capital of the United States of America.
       The capital of the United States of America is Washington
    ===============================
    Prompt: The future of AI is
    Generated text:  full of exciting possibilities, but it's not a mystery. AI has the potential to transform not just the way we live and work, but the way we interact with each other. That's why it's so important to understand the potential of AI, and to apply it to improve people's lives.
    
    AI is revolutionizing industries like healthcare, finance, transportation, and manufacturing, among others. It's also improving our lives in ways we never imagined possible. For example, AI-powered virtual assistants can help us find the information we need quickly and accurately. They can also help us communicate with our loved ones and families in new and exciting ways.
    
    


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your career and interests. Let's chat about [Your Area of Expertise] and see how we can work together to achieve [Your Goal]. I look forward to hearing from you! [Name] [Company Name] [Job Title] [Company Address] [City, State, Zip Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Website URL] [Company Website] [Company Social Media] [Company Blog] [Company Podcast] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. The city is home to many cultural institutions, including the Louvre Museum, the Musée d'Orsay, and the Centre Pompidou. Paris is a popular tourist destination, with millions of visitors each
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI will continue to automate tasks that are currently performed by humans, such as data analysis, decision-making, and routine maintenance. This will lead to increased efficiency and productivity, but it will also create new job opportunities.
    
    2. Enhanced human-computer interaction: AI will continue to improve its ability to understand and respond to human language, emotions, and intentions. This will lead to more natural and intuitive interactions between humans and AI systems.
    
    3. AI will become more integrated with other technologies: AI will continue to be integrated with other technologies, such as the Internet of Things (
    


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
    Generated text:  [Name]. I am [Age]. I come from [Country]. I currently [job title or role] at [Company]. What is your occupation? [Name] is passionate about [mention a hobby, interest or career]. How do you stay motivated during hard times? I find my motivation comes from [mention how you stay focused and stay dedicated in your work]. Where do you come from? I come from [country]. What inspired you to pursue a career in [career field]? [Name] was born in [birth country] and grew up in [location]. What is your favorite [occupation or hobby? ]? It is
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is the largest and most famous city in France, located on the left bank of the Seine River. It is an important cultural, artistic, and historical center, known for its extensive museums, landmarks, and annual festivals such as the Eiffel Tower parade and the annual Le Marseillais market. Paris is also known for its cuisine, fashion, and its role in shaping modern French identity. The city is home to many renowned museums, including the Louvre and the Musée d'Orsay, and its iconic landmarks include the Eiffel Tower, Notre-Dame Cathedral, and the Arc de Triomp
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but it is clear that it will continue to evolve and change significantly in the coming years. Here are some possible trends that could shape the future of AI:
    
    1. Increased focus on ethical and social implications: As more and more AI is being developed, there is a growing concern about the potential impact on society. The future of AI will likely be shaped by a growing awareness of the ethical and social implications of AI, including issues such as bias, privacy, and security.
    
    2. Greater use of AI in healthcare and medicine: As more AI is developed, there is a growing recognition of the potential to use AI to improve healthcare


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

     ______

    _.

     I

     am

     an

     AI

     language

     model

     designed

     to

     assist

     with

     various

     tasks

    .

     How

     can

     I

     assist

     you

     today

    ?

     Let

     me

     know

     if

     you

     need

     any

     information

     or

     have

     a

     question

    .

     I

     am

     here

     to

     help

     you

     with

     your

     queries

    .

     Just

     let

     me

     know

     what

     you

     need

    ,

     and

     I

    'll

     do

     my

     best

     to

     provide

     you

     with

     the

     information

     you

    're

     looking

     for

    .

     How

     can

     I

     assist

     you

     today

    ?

     I

    'm

     here

     to

     help

     you

     with

     your

     queries

    .

     Just

     let

     me

     know

     what

     you

     need

    ,

     and

     I

    'll

     do

     my

     best

     to

     provide

     you

     with

     the

     information

     you

    're

     looking

     for

    .

     How

     can

     I

     assist

     you

     today

    ?

     I

    'm

     here

     to

     help

     you

     with

     your

     queries

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     as

     the

     "

    City

     of

     Love

    "

     and

     is

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

    ,

     as

     well

     as

     numerous

     other

     historic

     landmarks

     and

     museums

    .

     Paris

     is

     a

     vibrant

     and

     multicultural

     city

     with

     a

     diverse

     range

     of

     cultures

    ,

     including

     French

    ,

     Italian

    ,

     and

     Jewish

    .

     It

     is

     also

     known

     for

     its

     rich

     history

    ,

     including

     many

     famous

     monuments

     and

     events

    ,

     such

     as

     the

     Bast

    ille

     Day

     celebrations

     in

     May

     and

     the

     French

     New

     Year

     celebrations

     in

     December

    .

     The

     city

     is

     also

     known

     for

     its

     modern

     architecture

     and

     electronic

     music

     scene

    .

     Overall

    ,

     Paris

     is

     a

     bustling

     and

     exciting

     city

     that

     is

     an

     important

     cultural

     and

     historical

     center

     for

     France

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     unpredictable

     and

     there

     are

     many

     possible

     trends

     that

     could

     impact

     the

     field

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     in

     healthcare

    :

     With

     the

     prevalence

     of

     AI

     in

     healthcare

    ,

     it

     is

     likely

     that

     more

     medical

     professionals

     and

     doctors

     will

     use

     AI

     to

     assist

     in

     diagnosis

     and

     treatment

    .

     This

     could

     lead

     to

     more

     accurate

     and

     precise

     diagnoses

    ,

     and

     ultimately

    ,

     improved

     patient

     outcomes

    .
    


    2

    .

     AI

     in

     finance

    :

     AI

     is

     already

     being

     used

     in

     financial

     services

     to

     automate

     trading

    ,

     risk

     management

    ,

     and

     fraud

     detection

    .

     As

     AI

     technology

     continues

     to

     improve

    ,

     it

     is

     likely

     that

     it

     will

     become

     more

     integrated

     into

     the

     financial

     industry

     as

    



```python
llm.shutdown()
```
