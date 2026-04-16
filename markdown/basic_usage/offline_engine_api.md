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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-16 09:35:31] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.23it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.23it/s]


    2026-04-16 09:35:35,172 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 09:35:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  4.93it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  4.93it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  4.93it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:09,  4.93it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:09,  4.93it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:03<00:09,  4.93it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:03<00:09,  4.93it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:03<00:09,  4.93it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:03<00:09,  4.93it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:03<00:03, 11.23it/s] 

    Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:03<00:03, 11.23it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:02, 16.46it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:02, 16.46it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:02, 16.46it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:02, 16.46it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:02, 16.46it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:02, 16.46it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:03<00:02, 16.46it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:03<00:02, 16.46it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]

    Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 23.35it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 36.99it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 36.99it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 36.99it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 36.99it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 36.99it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 36.99it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 36.99it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:03<00:00, 36.99it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 42.52it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 42.52it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 42.52it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 42.52it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 42.52it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 42.52it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 42.52it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 42.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.58 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.56 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.56 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.55 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.55 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.55 GB):   7%|▋         | 4/58 [00:00<00:03, 13.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.55 GB):   7%|▋         | 4/58 [00:00<00:03, 13.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.55 GB):   7%|▋         | 4/58 [00:00<00:03, 13.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.55 GB):   7%|▋         | 4/58 [00:00<00:03, 13.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.55 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.54 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.54 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.97it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.54 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.53 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.53 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.53 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.06it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.53 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.53 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.52 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.52 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.06it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=71.52 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.52 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.51 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.51 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.50 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.48 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.48 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.89it/s]Capturing num tokens (num_tokens=960 avail_mem=71.50 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.89it/s] Capturing num tokens (num_tokens=896 avail_mem=71.50 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.89it/s]Capturing num tokens (num_tokens=832 avail_mem=71.49 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.89it/s]Capturing num tokens (num_tokens=768 avail_mem=71.49 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.89it/s]

    Capturing num tokens (num_tokens=704 avail_mem=71.49 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.89it/s]Capturing num tokens (num_tokens=704 avail_mem=71.49 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.45it/s]Capturing num tokens (num_tokens=640 avail_mem=71.48 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.45it/s]Capturing num tokens (num_tokens=576 avail_mem=71.48 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.45it/s]Capturing num tokens (num_tokens=512 avail_mem=71.47 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.45it/s]Capturing num tokens (num_tokens=480 avail_mem=71.49 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.45it/s]Capturing num tokens (num_tokens=448 avail_mem=71.48 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.45it/s]Capturing num tokens (num_tokens=448 avail_mem=71.48 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=416 avail_mem=71.48 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=384 avail_mem=71.48 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=352 avail_mem=71.48 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.23it/s]

    Capturing num tokens (num_tokens=320 avail_mem=71.47 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=288 avail_mem=71.47 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.23it/s]Capturing num tokens (num_tokens=288 avail_mem=71.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.97it/s]Capturing num tokens (num_tokens=256 avail_mem=71.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.97it/s]Capturing num tokens (num_tokens=240 avail_mem=71.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.97it/s]Capturing num tokens (num_tokens=224 avail_mem=71.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.97it/s]Capturing num tokens (num_tokens=208 avail_mem=71.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.97it/s]Capturing num tokens (num_tokens=192 avail_mem=71.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.97it/s]Capturing num tokens (num_tokens=192 avail_mem=71.46 GB):  71%|███████   | 41/58 [00:01<00:00, 40.44it/s]Capturing num tokens (num_tokens=176 avail_mem=71.45 GB):  71%|███████   | 41/58 [00:01<00:00, 40.44it/s]Capturing num tokens (num_tokens=160 avail_mem=71.45 GB):  71%|███████   | 41/58 [00:01<00:00, 40.44it/s]

    Capturing num tokens (num_tokens=144 avail_mem=71.44 GB):  71%|███████   | 41/58 [00:01<00:00, 40.44it/s]Capturing num tokens (num_tokens=128 avail_mem=71.44 GB):  71%|███████   | 41/58 [00:01<00:00, 40.44it/s]Capturing num tokens (num_tokens=112 avail_mem=71.44 GB):  71%|███████   | 41/58 [00:01<00:00, 40.44it/s]Capturing num tokens (num_tokens=112 avail_mem=71.44 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=96 avail_mem=71.44 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.67it/s] Capturing num tokens (num_tokens=80 avail_mem=71.43 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=64 avail_mem=71.43 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=48 avail_mem=71.43 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=32 avail_mem=71.42 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.67it/s]

    Capturing num tokens (num_tokens=32 avail_mem=71.42 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=28 avail_mem=71.42 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=24 avail_mem=71.41 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=20 avail_mem=71.41 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=16 avail_mem=71.41 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=16 avail_mem=71.41 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.01it/s]Capturing num tokens (num_tokens=12 avail_mem=71.40 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.01it/s]Capturing num tokens (num_tokens=8 avail_mem=71.40 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.01it/s] Capturing num tokens (num_tokens=4 avail_mem=71.40 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.01it/s]Capturing num tokens (num_tokens=4 avail_mem=71.40 GB): 100%|██████████| 58/58 [00:01<00:00, 33.31it/s]


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
    Generated text:  John. I have traveled around the world and have lived in different places. One of the most interesting places I visited was Scotland. 
    
    Scotland is the 11th largest country in the world, and it is made up of five main islands, four smaller islands, and many smaller islets and peninsulas. 
    
    Scotland is known for its whisky, which is the best-selling spirit in the world, and is also famous for its woolen cloth and tartan. The Scottish Highlands, a highland area, is a place of striking beauty. 
    
    My first day in Scotland, I started my day by going to a local pub
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. Does this mean that the United States has no president? While it is a fact that the president of the United States is a man, it is not a fact that the United States has no president. There have been presidents in the history of the United States, even though they have not been a man.
    The President of the United States is elected through a process of voting and election, and men are elected as the President. The United States has had male presidents since its founding, with the exception of the first five presidents, who were all women. However, the country's history of female presidents is a long one, and
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which has a population of approximately 2.8 million. What is the population density in Paris?
    To calculate the population density in Paris, we need to divide the total population by the total area of the city.
    Assuming that the city is a perfect square, we can find the side length of the square by taking the square root of the total area.
    The side length of the square is approximately 25 km, since 2.8 million divided by 10,000 is approximately 28,000.
    The population density can then be calculated as the population divided by the side length of
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be diverse, complex and fast moving, which means AI needs to be more than just a tool for our industries. To achieve this, it is necessary to introduce these skills.
    To achieve this, it is necessary to introduce these skills.
    AI is changing the way we live our lives. It is making our lives easier, safer and more efficient. If we want to keep up with this, we must be able to use it.
    To this end, the ability to understand, interpret and use AI is becoming increasingly important in order to be able to benefit from the benefits of AI. This means that in order to be able to use


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a unique skill or trait here]. And what can you tell me about your work at [company name]? I'm a [insert a unique skill or trait here]. And what can you tell me about your hobbies or interests? I'm a [insert a unique skill or trait here]. And what can you tell me about your personal life? I'm a [insert a unique skill or trait here]. And what can you tell me about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a bustling metropolis with a rich cultural heritage and is a popular tourist destination. It is also known for its cuisine, including French cuisine, and its fashion industry. The city is home to many international organizations and is a major center for business, politics, and culture in France. Paris is a city of contrasts, with its historical architecture and modern skyscrapers, and its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to improve, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will lead to increased efficiency and productivity, but it will also create new jobs and require new skills.
    
    2. AI ethics and privacy: As AI technology becomes more advanced, there will be a growing concern about its impact on society. This includes issues such as bias in algorithms, privacy concerns, and the potential for AI
    


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
    Generated text:  [Your Name] and I am a [Your Profession] from [Your Country]. I have been programming and developing software since I was a [Your Age/Stage] and have been working on [Your Special Interest/Special Skill] for a long time. I am passionate about [Your Passion], and I believe that my work is making a positive impact on [Your Industry/Field]. I have always been a [Your Personality Trait], and I am always striving to improve myself, no matter how challenging the task may be. I love [Your Hobby], and I am looking forward to [Your Next Goal] in the near future.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    The statement is: Paris, France's capital, is located in the northwestern part of the country. Its climate is subtropical, with mild winters and hot summers. It is a city with a mix of historical and modern influences, known for its rich cultural heritage, vibrant arts scene, and a world-renowned neoclassical architecture. It is often referred to as the "City of Love" due to its romantic and romantic architecture. Paris is also home to numerous museums, galleries, and historical sites, including the Louvre, Notre-Dame Cathedral, and the Champs-Élysées. It's a popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities, and there are several potential trends that could shape its development. Here are some possible future trends in artificial intelligence:
    
    1. Personalization: As AI technologies become more advanced, the ability to personalize experiences will become more common. This could involve using AI to tailor content, recommendations, and even marketing to each individual user, leading to more tailored and relevant experiences.
    
    2. Autonomous robots: Robots that are autonomous and can make decisions based on pre-programmed algorithms will become increasingly common. This could lead to the development of robots that can perform tasks such as cleaning, gardening, or transportation, in addition to being able to work


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

     am

     a

     [

    Your

     profession

     or

     occupation

    ]

     with

     a

     passion

     for

     [

    Your

     hobbies

     or

     interests

    ].

     I

     am

     a

     [

    Your

     background

    ]

     with

     a

     deep

     understanding

     of

     [

    Your

     subject

     or

     area

     of

     interest

    ].

     I

     love

     to

     [

    Your

     hobbies

     or

     activities

    ],

     and

     I

     am

     always

     looking

     for

     ways

     to

     [

    Your

     goals

     or

     interests

    ].

     I

     am

     [

    Your

     age

     and

     gender

    ]

     and

     I

     am

     [

    Your

     personality

     traits

     or

     qualities

    ].

     I

     am

     always

     [

    Your

     positive

     traits

     or

     qualities

    ].

     I

     am

     always

     ready

     to

     help

     and

     share

     my

     knowledge

    ,

     and

     I

     believe

     that

     everyone

     can

     learn

     and

     grow

     from

     their

     mistakes

    .

     Welcome

     to

     my

     world

     and

     I

     look

     forward

     to

    
    
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

     city

     of

     France

    ."

     
    


    This

     brief

     statement

     con

    veys

     the

     main

     fact

     about

     Paris

     being

     the

     capital

     of

     France

    ,

     meeting

     the

     requirement

     of

     being

     a

     factual

     statement

    .

     However

    ,

     a

     more

     complete

     statement

     might

     be

    :

     "

    Paris

     is

     the

     largest

     and

     most

     important

     city

     in

     France

    ,

     serving

     as

     the

     capital

     of

     France

    ."

     This

     would

     provide

     a

     slightly

     more

     detailed

     explanation

     of

     what

     Paris

     is

    ,

     giving

     readers

     a

     clearer

     understanding

     of

     the

     city

    's

     significance

    .

     
    


    The

     second

     statement

     "

    Paris

     is

     the

     capital

     city

     of

     France

    "

     is

     more

     concise

     but

     still

     includes

     the

     main

     point

     that

     Paris

     is

     the

     capital

    .

     
    


    Among

     the

     given

     options

    ,

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     poised

     to

     see

     significant

     changes

     as

     technology

     continues

     to

     evolve

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

     Increased

     integration

     with

     human

     decision

    -making

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     are

     likely

     to

     be

     integrated

     more

     deeply

     into

     the

     decision

    -making

     process

    .

     This

     could

     result

     in

     more

     accurate

     and

     nuanced

     AI

     systems

     that

     can

     help

     humans

     make

     better

    -in

    formed

     decisions

    .
    


    2

    .

     Improved

     natural

     language

     processing

    :

     Natural

     language

     processing

     (

    N

    LP

    )

     has

     become

     increasingly

     important

     in

     AI

    ,

     with

     applications

     ranging

     from

     chat

    bots

     to

     speech

     recognition

    .

     As

     N

    LP

     continues

     to

     advance

    ,

     AI

     systems

     will

     be

     able

     to

     better

     understand

     human

     language

     and

     respond

     appropriately

    .
    


    3

    .

     Enhanced

     machine

     learning

    :

    



```python
llm.shutdown()
```
