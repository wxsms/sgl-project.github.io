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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.33it/s]


    2026-04-14 16:04:32,222 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 16:04:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.70it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.05it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.05it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.05it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.05it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.05it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.05it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.05it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.05it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.05it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.05it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.61it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.74it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.74it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.74it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.74it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.74it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.74it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.74it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.84it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.84it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.84it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.84it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.84it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.84it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.84it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.84it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 46.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:02, 18.67it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.34 GB):   7%|▋         | 4/58 [00:00<00:02, 18.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.31 GB):   7%|▋         | 4/58 [00:00<00:02, 18.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.02 GB):   7%|▋         | 4/58 [00:00<00:02, 18.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.02 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.01 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.01 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.01 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.37it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=119.00 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.00 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.00 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=119.00 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.00 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.99 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.99 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.99 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.00it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.00it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=118.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.51it/s]Capturing num tokens (num_tokens=960 avail_mem=118.97 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.51it/s] Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.51it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.51it/s]Capturing num tokens (num_tokens=768 avail_mem=118.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.51it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.51it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]

    Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.41it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.41it/s]Capturing num tokens (num_tokens=384 avail_mem=118.94 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.41it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.41it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.41it/s]Capturing num tokens (num_tokens=288 avail_mem=118.93 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.41it/s]Capturing num tokens (num_tokens=288 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.43it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.43it/s]

    Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.43it/s]Capturing num tokens (num_tokens=224 avail_mem=118.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.43it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.43it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.43it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  71%|███████   | 41/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  71%|███████   | 41/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=160 avail_mem=118.91 GB):  71%|███████   | 41/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  71%|███████   | 41/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  71%|███████   | 41/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=112 avail_mem=118.90 GB):  71%|███████   | 41/58 [00:01<00:00, 41.86it/s]

    Capturing num tokens (num_tokens=112 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.95it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.95it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.95it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.95it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.95it/s]Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.95it/s]Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.30it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.31it/s] Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 37.70it/s]


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
    Generated text:  Faniya. I'm 15 years old, and I come from Kharkov, Ukraine. I'm a student at the Karolinska University in Stockholm, Sweden. I'm a journalist and I'm working in the category of news.
    
    What are your hobbies? I love sports, especially tennis, basketball, and hockey. I like reading and watching movies. I also like to do yoga and meditate. I like to cook meals, go for walks, and listen to music.
    
    Tell me about your daily life. I usually wake up at 7:00 in the morning and then I have breakfast. I
    ===============================
    Prompt: The president of the United States is
    Generated text:  250 cm tall. If the average height of the president and Vice President is 180 cm, what is the total height of the two people?
    
    To determine the total height of the President and Vice President, we start with the given information:
    
    1. The height of the president: \(250 \text{ cm}\)
    2. The average height of the president and vice president: \(180 \text{ cm}\)
    
    First, we calculate the total height of the two people by adding their individual heights together:
    
    \[
    \text{Total height} = \text{height of the president}
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the 22nd largest city in the world and home to the European Parliament. What is the capital of Ireland?
    
    The capital of Ireland is Dublin. While Dublin is the capital of Ireland, Paris is the capital of France. Paris is the largest city in France, while Dublin is the capital of Ireland. Paris is known for its rich history, art, and cultural institutions, while Dublin is famous for its architecture, fashion, and various museums and theaters. Both cities are important economic and cultural hubs in their respective countries. 
    
    So, while Dublin is the capital of Ireland, Paris is the capital of France. Would
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about the speed and accuracy of its algorithms, but also about the emotional intelligence and empathy it brings to people. In this blog, we explore the intersection of AI and emotional intelligence and how it can enhance the efficiency and effectiveness of AI systems in various domains. We also discuss the ethical considerations that must be taken into account when designing AI systems, and how it can be used to create a more compassionate and empathetic society. So, let's dive in and find out how AI can be integrated with emotional intelligence in order to create a better future for all. 🌟 #AI #EmotionalIntelligence #FutureOfAI #


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art scene, and its role in the French Revolution. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. It is a popular tourist destination and a major economic center in Europe. The city is home to many famous museums, theaters, and restaurants, and is a major hub for international
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive systems, as well as more efficient and effective ways of interacting with humans.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment,
    


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
    Generated text:  _____. I'm ____ years old and ____ years strong. I enjoy reading books, playing sports, and watching movies. What's your favorite subject to learn? I'm a good listener, and I'm always trying to learn new things. What's your hobby or passion? I like to take photos, so I'm a photographer. I'm patient and have a great sense of humor. I'm not afraid to make mistakes, and I'm always ready to learn. How do you think you'll use this information to help a company? I'm a project manager. I know how to get things done. I can create a plan for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower and vibrant cultural scene. It serves as the political and economic center of the country, as well as the capital of France. The city is home to many historical landmarks, including the Louvre Museum, Notre-Dame Cathedral, and the Parc des Expositions. It's a bustling hub of activity during the summer and winter seasons, with many famous events taking place in the city during the year. Paris is also a popular tourist destination for tourists and locals alike. Overall, the city of Paris is a beautiful and vibrant metropolis with a rich cultural and historical legacy. Paris is a city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a field of opportunity and exploration, with a wide range of trends that are likely to shape its development. Here are some possible future trends in AI:
    
    1. Increased use of AI in healthcare: AI has already been used in healthcare to diagnose diseases, predict treatment outcomes, and manage medical data. As AI technology advances, we can expect to see increased use of AI in healthcare to improve patient outcomes and enhance the efficiency of healthcare delivery.
    
    2. Increased use of AI in consumer electronics: As the demand for high-quality, durable consumer electronics continues to grow, AI will play an increasingly important role in shaping the industry. AI-powered voice assistants,


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

     first

     name

    ]

     and

     I

    'm

     a

     [

    insert

     profession

    ,

     company

    ,

     or

     title

    ]

     with [

    insert

     any

     relevant

     experience

     or

     skills

    ].

     I

    'm

     passionate

     about

     [

    insert

     why

     you

     enjoy

     your

     job

     or

     what

     you

     enjoy

     about

     your

     profession

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     my

     skills

     and

     expertise

    .

     I

    'm

     also

     a

     [

    insert

     any

     interests

     or

     hobbies

    ],

     and

     I

     love

     [

    insert

     any

     hobbies

     you

     have

    ].

     I

    'm

     excited

     to

     bring

     my

     [

    insert

     skills

     or

     talents

    ]

     to

     anyone

     I

     meet

     and

     work

     hard

     to

     make

     a

     positive

     impact

    .

     Thank

     you

    !

     [

    Insert

     your

     name

     and

    /or

     title

    ].

     This

     is

     a

     bit

     vague

    ,

     so

     I

    'll

     make

     it

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     statement

     is

    :


    "

    Paris

     is

     the

     capital

     of

     France

    ."

     
    


    This

     statement

     accurately

     reflects

     the

     fact

     that

     Paris

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     European

     Union

    ,

     serving

     as

     the

     officially

     designated

     capital

     of

     France

    .

     It

     is

     the

     center

     of

     French

     politics

    ,

     culture

    ,

     and

     society

    .

     
    


    For

     a

     more

     detailed

     analysis

    ,

     consider

     the

     following

     points

    :
    


    1

    .

     Ge

    ographical

     significance

    :

     Paris

     is

     the

     largest

     city

     in

     France

    ,

     followed

     by

     Lyon

    ,

     Marseille

    ,

     and

     L

    ille

    .


    2

    .

     Cultural

     importance

    :

     Known

     for

     its

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     various

     museums

     and

     art

     galleries

    .


    3

    .

     Political

     role

    :

     As

     the

     seat

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     multitude

     of

     developments

    ,

     including

    :
    


    1

    .

     Increased

     automation

     and

     robotics

    :

     AI

     will

     continue

     to

     automate

     many

     tasks

    ,

     from

     manufacturing

     to

     customer

     service

    .

     This

     will

     require

     more

     human

     intervention

    ,

     but

     it

     will

     also

     create

     new

     roles

     and

     jobs

     that

     can

     be

     designed

     and

     built

     with

     AI

    .
    


    2

    .

     Improved

     AI

     algorithms

    :

     As

     AI

     becomes

     more

     sophisticated,

     it

     will

     become

     better

     at

     understanding

     and

     predicting

     complex

     patterns

     in

     data

    .

     This

     will

     lead

     to

     more

     accurate

     predictions

     and

     better

     decision

    -making

    .
    


    3

    .

     Adv

    ancements

     in

     AI

     ethics

    :

     As

     AI

     becomes

     more

     advanced

    ,

     there

     will

     be

     a

     greater

     focus

     on

     ethical

     considerations

    .

     This

     will

     include

     issues

     such

     as

     bias

    ,

     transparency

    ,

     and

    



```python
llm.shutdown()
```
