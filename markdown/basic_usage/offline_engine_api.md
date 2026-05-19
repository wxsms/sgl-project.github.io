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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.86it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.86it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.88it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.88it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.88it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.88it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.88it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.88it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.88it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.88it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.88it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.88it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 23.88it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 32.83it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 32.83it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 32.83it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 32.83it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 32.83it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 32.83it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.48it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.75 GB):   3%|▎         | 2/58 [00:00<00:03, 16.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:03, 16.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:03, 16.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:03, 16.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.74 GB):   9%|▊         | 5/58 [00:00<00:02, 20.43it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.74 GB):   9%|▊         | 5/58 [00:00<00:02, 20.43it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.73 GB):   9%|▊         | 5/58 [00:00<00:02, 20.43it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.72 GB):   9%|▊         | 5/58 [00:00<00:02, 20.43it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=73.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.72 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.71 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.71 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.71 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.71 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.40it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=73.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.69 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.69 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.69 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.36it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=73.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.67 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=960 avail_mem=73.68 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.62it/s] Capturing num tokens (num_tokens=896 avail_mem=73.68 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=832 avail_mem=73.68 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=768 avail_mem=73.67 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=768 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:01, 30.35it/s]Capturing num tokens (num_tokens=704 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:00<00:01, 30.35it/s]Capturing num tokens (num_tokens=640 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.35it/s]Capturing num tokens (num_tokens=576 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.35it/s]

    Capturing num tokens (num_tokens=512 avail_mem=73.65 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.35it/s]Capturing num tokens (num_tokens=480 avail_mem=73.67 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.35it/s]Capturing num tokens (num_tokens=480 avail_mem=73.67 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=448 avail_mem=73.66 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=416 avail_mem=73.66 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=384 avail_mem=73.66 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=352 avail_mem=73.65 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=352 avail_mem=73.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=320 avail_mem=73.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=288 avail_mem=73.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.24it/s]

    Capturing num tokens (num_tokens=256 avail_mem=73.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=240 avail_mem=73.64 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=224 avail_mem=73.64 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.24it/s]Capturing num tokens (num_tokens=224 avail_mem=73.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=208 avail_mem=73.63 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=192 avail_mem=73.63 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=176 avail_mem=73.63 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=160 avail_mem=73.63 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=144 avail_mem=73.62 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.03it/s]

    Capturing num tokens (num_tokens=144 avail_mem=73.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=128 avail_mem=73.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=112 avail_mem=73.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=96 avail_mem=73.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.13it/s] Capturing num tokens (num_tokens=80 avail_mem=73.61 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=80 avail_mem=73.61 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=64 avail_mem=73.61 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.03it/s]

    Capturing num tokens (num_tokens=48 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=32 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=28 avail_mem=73.60 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=28 avail_mem=73.60 GB):  90%|████████▉ | 52/58 [00:01<00:00, 28.97it/s]Capturing num tokens (num_tokens=24 avail_mem=73.59 GB):  90%|████████▉ | 52/58 [00:01<00:00, 28.97it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  90%|████████▉ | 52/58 [00:01<00:00, 28.97it/s]

    Capturing num tokens (num_tokens=16 avail_mem=73.32 GB):  90%|████████▉ | 52/58 [00:01<00:00, 28.97it/s]Capturing num tokens (num_tokens=12 avail_mem=73.31 GB):  90%|████████▉ | 52/58 [00:01<00:00, 28.97it/s]Capturing num tokens (num_tokens=12 avail_mem=73.31 GB):  97%|█████████▋| 56/58 [00:01<00:00, 29.76it/s]Capturing num tokens (num_tokens=8 avail_mem=72.92 GB):  97%|█████████▋| 56/58 [00:01<00:00, 29.76it/s] Capturing num tokens (num_tokens=4 avail_mem=72.60 GB):  97%|█████████▋| 56/58 [00:01<00:00, 29.76it/s]Capturing num tokens (num_tokens=4 avail_mem=72.60 GB): 100%|██████████| 58/58 [00:01<00:00, 29.16it/s]


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
    Generated text:  Hanh. I'm a Chinese boy. I have a new pen friend. She is from France. Her name is Sarah. We have the same birthday. She likes cakes and Italian food. She lives in a city far away. She writes to me every day. I tell her about my favorite TV shows and movies. She tells me about her trip to Paris, and the food there. We share many things. We both like to eat hamburgers. The most interesting part is that she said that she was born in a barn and wrote me her birthday greeting card. How do you say it in English?
    
    To say "how do
    ===============================
    Prompt: The president of the United States is
    Generated text:  3 feet tall. If a woman is 5 feet 4 inches tall and a man is 6 feet tall, how many feet taller is the president than the man? To determine how many feet taller the president is than the man, we need to follow these steps:
    
    1. Convert the height of the woman from feet and inches to just feet.
    2. Convert the height of the man from feet and inches to just feet.
    3. Subtract the height of the man from the height of the president to find the difference in feet.
    
    First, let's convert the heights of the women and the man to just feet:
    - The
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Marseille
    C. Nice
    D. Lyon
    Answer:
    
    A
    
    [Multiple Choice Question] (3 points) At the Fourth Plenary Session of the 19th Central Committee of the Communist Party of China, General Secretary Xi Jinping pointed out that the 19th National Congress of the Communist Party of China, known as the ____, was convened at a significant time when the Communist Party of China has entered a new era of socialism with Chinese characteristics, and the new era of socialism with Chinese characteristics has entered a stage of fully building a modern socialist country and the first centenary goal has
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but the path to it is fraught with challenges. How can we ensure that we create a safe, ethical, and sustainable future where AI is used in a responsible and beneficial way? Here are some key points to consider:
    
    1. Develop regulations: Governments should develop regulations to ensure that AI is used in a responsible and ethical manner. This includes ensuring that AI is developed and used in a way that protects privacy and security, and that it is used in a way that is fair and transparent.
    
    2. Encourage collaboration: Collaboration between industries, academia, and government can help ensure that AI is used in a responsible and beneficial way.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your personality or skills]. What do you enjoy doing in your free time? I enjoy [insert a hobby or activity that you enjoy doing]. What's your favorite book or movie? I love [insert a favorite book or movie]. What's your favorite hobby or activity? I love [insert a hobby or activity that you enjoy doing]. What's your favorite place to go? I love [insert a favorite place
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination. The city is known for its cuisine, fashion, and music, and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its elegant architecture, vibrant nightlife, and diverse cultural scene. Its status as the capital of France is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn from and adapt to the behavior and preferences of humans.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be a need to address privacy and security concerns. This may lead to new regulations and technologies to protect user data.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, and there is potential
    


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
    Generated text:  [Your Name], and I am [Your Profession]. I have always been passionate about [Your Profession], and I believe that every person should have the opportunity to succeed in their field. I am constantly seeking to learn new things and improve myself, and I am always looking for opportunities to contribute to my community and help others. Please let me know if you would like to meet me. Let me know if there is anything I can do to make you feel welcome and comfortable around me. [Your Name] [Your Profession] [Your profession] [Your hobbies and interests] [Your career goals and aspirations] [Your education and training]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the most populous city in France and the third-most populous city in the European Union after London and Madrid.
    How to prepare for a job interview in France: A guide. How to prepare for a job interview in France: A guide.
    To help prepare for a job interview in France, it is important to research the company and their products or services. You should also research the company's history, management team, and culture. It is important to be well-prepared for the interview, including any relevant personal or professional experiences. It is also important to be comfortable with the language and culture of the country and company. Finally
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting, with many possibilities on the horizon. Here are some potential trends in AI that are currently being explored:
    
    1. Self-learning AI: AI will become more self-learning, with the ability to learn and improve over time. This will allow for more efficient and accurate solutions to complex problems.
    
    2. Explainable AI: AI systems will become more transparent and explainable, with the ability to provide explanations for their decisions. This will make it easier for humans to understand and trust the AI's output.
    
    3. Autonomous robots: The development of autonomous robots will continue to advance, with the ability to work in various environments, such as factories,


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

    First

     Name

    ]

     [

    Last

     Name

    ],

     and

     I

    ’m

     a

     [

    Your

     Genre

    ,

     if

     applicable

    ]

     author

    .

     I

     started

     my

     journey

     into

     the

     world

     of

     literature

     when

     I

     was

     just

     a

     child

    ,

     and

     I

    ’ve

     been

     fascinated

     by

     the

     stories

     that

     come

     to

     me

    .

     Over

     the

     years

    ,

     I

    ’ve

     had

     the

     privilege

     of

     reading

     countless

     books

    ,

     and

     I

    ’ve

     learned

     so

     much

     from

     them

    .

     I

    ’m

     now

     ready

     to

     take

     on

     the

     world

     and

     share

     my

     own

     stories

    ,

     just

     as

     I

     did

     when

     I

     was

     younger

    .

     Thank

     you

     for

     having

     me

    .

     To

     sum

     up

    ,

     I

    'm

     a

     [

    Your

     Character

    ]

     author

    .

     What

    's

     your

     favorite

     book

     to

     read

    ?

     As

     an

     AI

     language

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     stunning

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     culture

    .

     The

     city

     is home

     to

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

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     famous

     for

     its

     French

     cuisine

    ,

     fashion

    ,

     and

     world

    -ren

    owned

     film

     and

     music

     scenes

    .

     The

     city

     is

     a

     major

     hub

     for

     education

    ,

     science

    ,

     and

     culture

    ,

     with

     many

     famous

     universities

    ,

     theaters

    ,

     and

     museums

     located

     there

    .

     Its

     

    2

    0

    2

    0

     Population

     at

     the

     Time

     of

     the

     Paris

     Saint

    -G

    er

    main

     F

    OP

    L

     Cup

     Final

     played

     is

     

    8

    5

    8

    ,

    3

    2

    8

    .

     According

     to

     data

     from

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     trends

     and

     developments

    ,

     including

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     cloud

     computing

    ,

     robotics

    ,

     and

     smart

     homes

    ,

     we

     can

     expect

     to

     see

     more

     seamless

     integration

     and

     automation

     of

     AI

    -powered

     systems

    .
    


    2

    .

     Greater

     reliance

     on

     human

     oversight

    :

     AI

     systems

     will

     likely

     require

     human

     oversight

     to

     ensure

     that

     they

     are

     making

     ethical

     and

     safe

     decisions

    .

     This

     will

     lead

     to

     greater

     emphasis

     on

     human

    -centered

     design

     and

     greater

     integration

     of

     ethical

     considerations

     into

     AI

     development

    .
    


    3

    .

     Faster

     development

     and

     deployment

    :

     As

     AI

     technology

     advances

     at

     an

     unprecedented

     pace

    ,

     we

     can

     expect

     to

     see

     faster

     development

     and

     deployment

    



```python
llm.shutdown()
```
