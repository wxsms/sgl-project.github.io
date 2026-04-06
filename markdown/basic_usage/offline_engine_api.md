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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.10it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.08it/s]


    2026-04-06 06:12:27,787 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 06:12:27] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:07,  5.98it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:07,  5.98it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  5.98it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  5.98it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  5.98it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  5.98it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  5.98it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  5.98it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  5.98it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.13it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.09it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.09it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.09it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.09it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.09it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.09it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.09it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.09it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.63it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.63it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.63it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.63it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.63it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.63it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.63it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.67it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.67it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.67it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.67it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.67it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.67it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.67it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.66it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.66it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.66it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.66it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.66it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.66it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.66it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=121.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=3584 avail_mem=121.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=3328 avail_mem=121.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.80it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.98 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.98 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.98 GB):  31%|███       | 18/58 [00:00<00:01, 35.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.98 GB):  31%|███       | 18/58 [00:00<00:01, 35.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.97 GB):  31%|███       | 18/58 [00:00<00:01, 35.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.95 GB):  31%|███       | 18/58 [00:00<00:01, 35.22it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.96 GB):  31%|███       | 18/58 [00:00<00:01, 35.22it/s] Capturing num tokens (num_tokens=896 avail_mem=120.96 GB):  31%|███       | 18/58 [00:00<00:01, 35.22it/s]Capturing num tokens (num_tokens=896 avail_mem=120.96 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.04it/s]Capturing num tokens (num_tokens=832 avail_mem=120.96 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.04it/s]Capturing num tokens (num_tokens=768 avail_mem=120.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.04it/s]Capturing num tokens (num_tokens=704 avail_mem=120.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.04it/s]Capturing num tokens (num_tokens=640 avail_mem=120.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.04it/s]Capturing num tokens (num_tokens=576 avail_mem=120.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.04it/s]Capturing num tokens (num_tokens=576 avail_mem=120.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=512 avail_mem=120.94 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=480 avail_mem=120.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=448 avail_mem=120.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]

    Capturing num tokens (num_tokens=416 avail_mem=120.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=384 avail_mem=120.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=384 avail_mem=120.95 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=352 avail_mem=120.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=320 avail_mem=120.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=288 avail_mem=120.93 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=256 avail_mem=120.93 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=240 avail_mem=120.33 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=240 avail_mem=120.33 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=224 avail_mem=119.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=208 avail_mem=119.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=192 avail_mem=119.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.95it/s]

    Capturing num tokens (num_tokens=176 avail_mem=119.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=160 avail_mem=119.55 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=160 avail_mem=119.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s]Capturing num tokens (num_tokens=144 avail_mem=119.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s]Capturing num tokens (num_tokens=128 avail_mem=119.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s]Capturing num tokens (num_tokens=112 avail_mem=119.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s]Capturing num tokens (num_tokens=96 avail_mem=119.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s] Capturing num tokens (num_tokens=80 avail_mem=119.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.98it/s]Capturing num tokens (num_tokens=80 avail_mem=119.54 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=64 avail_mem=119.53 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=48 avail_mem=119.53 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=32 avail_mem=119.53 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.40it/s]

    Capturing num tokens (num_tokens=28 avail_mem=119.52 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=24 avail_mem=119.52 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=24 avail_mem=119.52 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=20 avail_mem=119.52 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=16 avail_mem=119.52 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=12 avail_mem=119.51 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=8 avail_mem=119.51 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.85it/s] Capturing num tokens (num_tokens=4 avail_mem=119.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=4 avail_mem=119.50 GB): 100%|██████████| 58/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=4 avail_mem=119.50 GB): 100%|██████████| 58/58 [00:01<00:00, 39.87it/s]


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
    Generated text:  Tom and I am a 14-year-old boy from the UK. I am here to answer your questions regarding the study of a certain subject. Let's go to our topic of "Mental Health". That means your mental health. The topic of mental health is important because it is important to know about mental health to be able to help others and also to be able to help ourselves. If we do not take care of our mental health, it will affect us negatively, both physically and emotionally. That is why it is important to know how to take care of our mental health. I want to talk about how to take care of
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to reduce the amount of global warming by reducing the amount of fossil fuels that are used. To achieve this, the president plans to invest in the development of a new kind of fuel called "Green Energy" which will cost $30 billion to develop and $12 billion in production costs per year. The president also plans to invest in the development of a new kind of fuel called "Eco Fuel" which will cost $20 billion to develop and $8 billion in production costs per year. The president plans to make profits on the new fuels by selling them for $50 per gallon, but it will take 5
    ===============================
    Prompt: The capital of France is
    Generated text:  located in which country?
    The capital of France is Paris, which is located in France. France is a country located in western Europe, bordering the Atlantic Ocean to the east and the Mediterranean Sea to the west. The country has a rich history and culture, and its capital city, Paris, is a hub for politics, culture, and arts. The capital of France is considered the "backbone" of the country, and it is the political, economic, and cultural center of the nation. Paris is also known for its fashion industry, art scene, and modern art museum. 
    
    The other countries in Europe that are located in the
    ===============================
    Prompt: The future of AI is
    Generated text:  bright! In the next decade, AI will become the dominant technological force, capable of solving some of the world’s most complex problems and helping us in our everyday lives. However, as AI advances and becomes more accessible, its impact on society and our health will also grow. This article explores the potential future of AI and how it will shape society in the coming years. We will examine the role of AI in medicine, education, and the environment, and consider how AI will affect the workforce and the economy in the coming decade. We will also examine the ethical considerations surrounding AI and the potential consequences of its misuse.
    What will be the impact


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [job title] at [company name], and I have been working in [position] for [number of years] years. I enjoy [job title] because [reason for job title]. What's your background? I have a [number of years] years of experience in [industry], and I have a [number of years] years of experience in [industry]. I'm always looking for new opportunities to learn and grow. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, diverse culture, and vibrant nightlife. It is a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial general intelligence: As AI continues to improve, it is likely to become more capable of performing a wide range of tasks that were previously done by humans. This could lead to the development of more advanced AI systems that can perform tasks that were previously considered impossible for humans.
    
    2. Integration with human decision-making: AI is likely to become more integrated with human decision-making processes, allowing for more complex and nuanced decision-making. This could lead to more effective and
    


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
    Generated text:  [Name], I'm a [Type] [Occupation] and I'm currently [Current Location]. I've always been passionate about [Favorite Subject/Interest], and I'm dedicated to [Primary Goal/Responsibility]. I've been learning and growing in my career, always looking for ways to [Professional Development Goal/Responsibility]. I'm a [Hobby/Interest], and I enjoy [Specific Activity], which has been my way to unwind and relax. I believe in [Leadership Trait/Value], and I'm always eager to [Actions/Responsibilities/Goals/Goals]. What's your story? What's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, vibrant culture, and iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is the largest city in Europe and home to the European Parliament, the French Parliament, and the offices of the French President. Paris is also known for its cuisine, fashion, and annual festivals like the Louvre’s annual French New Year’s Eve fireworks display. Paris is a city of art, culture, and history, and is a must-visit for any traveler. The city is also home to a vibrant and diverse community of people, who celebrate French and international culture through
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a wide range of technological advances and applications. Some potential trends include:
    
    1. Increased use of AI for autonomous vehicles: As autonomous vehicles become more common, AI will play an increasingly important role in transportation and logistics. This could lead to a reduction in the need for human drivers and the development of self-driving cars that can navigate complex urban environments.
    
    2. AI for healthcare: AI will be used in healthcare to diagnose diseases, develop new treatments, and improve patient care. This could involve the use of AI-powered tools to analyze medical images, detect patterns in patient data, and identify genetic markers for disease.
    
    3. AI for


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

    Age

    ]

     year

     old

     [

    Gender

    ]

     from

     [

    Location

    ].

     I

    've

     always

     had

     an

     interest

     in

     technology

     and

     have

     been

     exploring

     the

     possibilities

     of

     it

     for

     many

     years

    .

     I

     have

     a

     passion

     for

     learning

     and

     continuously

     improve

     my

     skills

     in

     programming

     and

     [

    specific

     skill

    ].

     I

     enjoy

     brainstorm

    ing

     new

     ideas

     and

     developing

     innovative

     solutions

     for

     problems

    .

     I

     also

     have

     a

     keen

     sense

     of

     humor

     and

     enjoy

     making

     people

     laugh

    .

     What

     is

     the

     most

     interesting

     or

     unique

     aspect

     of

     your

     personality

    ?

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     feelings

     or

     personal

     experiences

     like

     humans

     do

    ,

     but

     I

     have

     learned

     a

     lot

     from

     the

     countless

     hours

     of

     programming

     that

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     

    2

    5

    th

     largest

     city

     in

     the

     world

     by

     population

    ,

     known

     for

     its

     historical

     significance

    ,

     museums

    ,

     and

     fashion

     industry

    .

     France

    's

     capital

     city

    ,

     Paris

    ,

     is

     the

     

    2

    5

    th

     largest

     city

     by

     population

    ,

     renowned

     for

     its

     rich

     history

    ,

     museums

    ,

     and

     fashion

     industry

    .

     The

     city

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     landmark

     landmarks

     like

     the

     Lou

    vre

     Museum

    ,

     and

     its

     vibrant

     nightlife

    .

     With

     its

     numerous

     museums

    ,

     art

     galleries

    ,

     and

     cultural

     institutions

    ,

     Paris

     is

     a

     cultural

     and

     historical

     gem

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

     also

     home

     to

     the

     iconic

     Lou

    vre

    ,

     which

     houses

     one

     of

     the

     largest

     art

     collections

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

     and

     exciting

    ,

     with

     many

     possibilities

     and

     potential

     applications

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Improved

     accuracy

    :

     One

     of

     the

     biggest

     benefits

     of

     AI

     is

     its

     ability

     to

     improve

     the

     accuracy

     of

     predictions

    .

     As

     AI

     algorithms

     become

     more

     sophisticated

    ,

     they

     can

     learn

     from

     large

     datasets

     and

     make

     better

     and

     more

     accurate

     predictions

    .
    


    2

    .

     Increased

     adoption

    :

     AI

     is

     becoming

     more

     accessible

     and

     affordable

    ,

     with

     more

     people

     being

     able

     to

     use

     it

     for

     various

     tasks

    .

     This

     increase

     in

     adoption

     could

     lead

     to

     more

     widespread

     use

     of

     AI

     in

     many

     industries

    .
    


    3

    .

     Greater

     integration

    :

     AI

     is

     already

     being

     integrated

     into

     many

     different

     areas

    ,

     such

     as

     healthcare

    ,

     finance

    



```python
llm.shutdown()
```
