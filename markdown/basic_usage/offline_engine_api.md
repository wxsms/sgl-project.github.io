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
    [2026-04-17 15:46:19] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.99it/s]


    2026-04-17 15:46:23,622 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 15:46:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.84it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.84it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.84it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.84it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.84it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:02<00:06,  6.84it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:02<00:06,  6.84it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:02<00:06,  6.84it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:02<00:06,  6.84it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.84it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.96it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.96it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 22.11it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 22.11it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 22.11it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 22.11it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 22.11it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 22.11it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 22.11it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 22.11it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 22.11it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 22.11it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 31.00it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 31.00it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 31.00it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 31.00it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 31.00it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 31.00it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 31.00it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 31.00it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 31.00it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 31.00it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.87it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 50.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=134.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=134.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=134.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=134.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=134.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=134.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=134.78 GB):   9%|▊         | 5/58 [00:00<00:02, 21.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=134.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=134.78 GB):   9%|▊         | 5/58 [00:00<00:02, 21.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=134.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=134.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=134.77 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=134.77 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=134.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=134.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.45it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=134.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=134.76 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=134.76 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=134.75 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=134.75 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=134.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=134.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=134.74 GB):  31%|███       | 18/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=134.74 GB):  31%|███       | 18/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1280 avail_mem=134.73 GB):  31%|███       | 18/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=1024 avail_mem=134.71 GB):  31%|███       | 18/58 [00:00<00:01, 35.29it/s]

    Capturing num tokens (num_tokens=960 avail_mem=134.73 GB):  31%|███       | 18/58 [00:00<00:01, 35.29it/s] Capturing num tokens (num_tokens=896 avail_mem=134.72 GB):  31%|███       | 18/58 [00:00<00:01, 35.29it/s]Capturing num tokens (num_tokens=896 avail_mem=134.72 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=832 avail_mem=134.72 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=768 avail_mem=134.72 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=704 avail_mem=134.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=640 avail_mem=134.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=640 avail_mem=134.71 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.32it/s]Capturing num tokens (num_tokens=576 avail_mem=134.71 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.32it/s]Capturing num tokens (num_tokens=512 avail_mem=134.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.32it/s]

    Capturing num tokens (num_tokens=480 avail_mem=134.71 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.32it/s]Capturing num tokens (num_tokens=448 avail_mem=134.71 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.32it/s]Capturing num tokens (num_tokens=448 avail_mem=134.71 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.21it/s]Capturing num tokens (num_tokens=416 avail_mem=134.71 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.21it/s]Capturing num tokens (num_tokens=384 avail_mem=134.71 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.21it/s]Capturing num tokens (num_tokens=352 avail_mem=134.70 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.21it/s]Capturing num tokens (num_tokens=320 avail_mem=134.70 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.21it/s]Capturing num tokens (num_tokens=320 avail_mem=134.70 GB):  60%|██████    | 35/58 [00:01<00:00, 36.56it/s]Capturing num tokens (num_tokens=288 avail_mem=134.70 GB):  60%|██████    | 35/58 [00:01<00:00, 36.56it/s]Capturing num tokens (num_tokens=256 avail_mem=134.69 GB):  60%|██████    | 35/58 [00:01<00:00, 36.56it/s]

    Capturing num tokens (num_tokens=240 avail_mem=134.69 GB):  60%|██████    | 35/58 [00:01<00:00, 36.56it/s]Capturing num tokens (num_tokens=224 avail_mem=134.69 GB):  60%|██████    | 35/58 [00:01<00:00, 36.56it/s]Capturing num tokens (num_tokens=224 avail_mem=134.69 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=208 avail_mem=134.68 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=192 avail_mem=134.68 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=176 avail_mem=134.68 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=160 avail_mem=134.68 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.27it/s]Capturing num tokens (num_tokens=160 avail_mem=134.68 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.08it/s]Capturing num tokens (num_tokens=144 avail_mem=134.67 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.08it/s]Capturing num tokens (num_tokens=128 avail_mem=134.67 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.08it/s]

    Capturing num tokens (num_tokens=112 avail_mem=134.67 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.08it/s]Capturing num tokens (num_tokens=96 avail_mem=134.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.08it/s] Capturing num tokens (num_tokens=96 avail_mem=134.66 GB):  81%|████████  | 47/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=80 avail_mem=134.66 GB):  81%|████████  | 47/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=64 avail_mem=134.66 GB):  81%|████████  | 47/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=48 avail_mem=134.65 GB):  81%|████████  | 47/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=32 avail_mem=134.65 GB):  81%|████████  | 47/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=32 avail_mem=134.65 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.76it/s]Capturing num tokens (num_tokens=28 avail_mem=134.64 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.76it/s]Capturing num tokens (num_tokens=24 avail_mem=134.64 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.76it/s]

    Capturing num tokens (num_tokens=20 avail_mem=134.64 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.76it/s]Capturing num tokens (num_tokens=16 avail_mem=134.64 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.76it/s]Capturing num tokens (num_tokens=16 avail_mem=134.64 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=12 avail_mem=134.63 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=8 avail_mem=134.63 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.95it/s] Capturing num tokens (num_tokens=4 avail_mem=134.63 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=4 avail_mem=134.63 GB): 100%|██████████| 58/58 [00:01<00:00, 34.82it/s]


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
    Generated text:  Ruth and I am a 14 year old girl. I have a goal of learning to ride a bicycle. I've been practicing riding it for a while, but I can't seem to get it right. I'm always trying to figure out what part of the bicycle is not supposed to be on the bicycle. 
    
    I have some questions about a part of the bicycle. What is the difference between a chain and a chain ring? Also, can I use the back of my hand to help me fix the bicycle?
    
    Thank you for your help!
    Ruth
    
    Ruth, congratulations on your goal to learn to ride a bicycle!
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. She is like the leader of the country. She is also the leader of the whole world. She has many important jobs to do every day. That's why she is very busy. She is very busy on the weekends. She spends much time in the meeting rooms. She talks to all the people who can be trusted to make the right decision. She talks to the leaders in all the countries of the world. She talks to the heads of companies. She speaks to the people who are working in the factories. She talks to the people who are in the schools. She talks to the people who
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Paris is located on the Île de France, an island lying on the English Channel, on the western coast of France.
    
    Paris is the third largest city in France by population, and the 5th largest city by area.
    
    The city is on the banks of the Seine river, which rises in the center of the city and flows for about 160 kilometers (100 miles) to the sea at the harbor.
    
    The Seine river forms the heart of Paris's waterway system, which includes the River Seine, the River Marne, the River Marne, and the River Orne.
    
    The
    ===============================
    Prompt: The future of AI is
    Generated text:  changing. In the last decade, we have seen a significant shift towards AI, which has transformed various industries and advanced the capabilities of AI models. However, the question remains – what will the future hold for AI? As AI is rapidly advancing, we can expect a wide range of possibilities, including new applications, improved accuracy, and more sophisticated models. Additionally, AI is not a one-size-fits-all solution, and as we continue to develop and improve our models, we may encounter new challenges and obstacles.
    One of the most significant challenges that we face in the future of AI is the issue of bias. Bias in AI models can lead


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and is home to many international organizations and institutions. The city is known for its rich history, including the influence of the French Revolution and the influence of the French language. It is also known for its cuisine, including French cuisine, and its fashion industry. Paris is a vibrant and dynamic city with a rich history and a diverse population. Its status as the capital of France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine to virtual assistants. Additionally, AI will likely continue to be used for ethical and social purposes, such as improving access to healthcare and reducing inequality. Finally, AI will likely continue to evolve and adapt to new challenges and opportunities, leading to new applications and breakthroughs in the field. Overall, the future of AI is likely to be one of continued innovation and growth.
    


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
    Generated text:  [Name], I'm a [Age] year old, [Occupation] with [Number of Employees] people. I work at [Location]. I love [Hobby/Activity/Interest] and I enjoy spending [Number of Hours] hours every day here. My favorite thing about [Location] is [Feature/Challenge], I've been here for [Number of Months] and I can't wait to see what [Next Milestone] will be. How about you, what's your name, age, occupation, location and hobby/interest? (Type your responses here) Hello, my name is [Name], I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France, located in the North West of the country. It is a cosmopolitan and diverse city with a rich history and culture. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and the Louvre Pyramid, and for its famous restaurants and cafes. Paris is also a major tourist destination, attracting millions of visitors each year. It is a city that is renowned for its art, music, and fashion, and is considered one of the most beautiful cities in the world. The city has a strong sense of community and a thriving
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and changing fast. Here are some possible trends in the AI field in the next few years:
    
    1. Increased Integration with Human Intelligence: AI is becoming more integrated with human intelligence, as machines are being trained on a wide range of data. This integration will lead to more effective human-AI collaboration and decision-making.
    
    2. Enhanced Perception and Understanding: AI will continue to develop advanced sensors and algorithms that allow it to perceive more effectively and understand more accurately.
    
    3. AI in Healthcare: AI is already revolutionizing healthcare, with advances in machine learning and natural language processing enabling more accurate diagnoses and personalized treatment plans. We can expect AI to


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

    insert

     occupation

    ].

     I

    'm

     a

     [

    insert

     your

     profession

    ]

     and

     I

    've

     always

     been

     passionate

     about

     [

    insert

     something

     you

     like

     doing

    ].

     I

     enjoy

     [

    insert

     something

     you

     do

     in

     your

     free

     time

    ].

     What

     would

     you

     like

     to

     know

     about

     me

    ?

     [

    Tell

     the

     interviewer

     your

     name

    ,

     occupation

    ,

     and

     a

     little

     bit

     about

     yourself

    .

     Start

     with

     a

     neutral

     introduction

    ,

     but

     don

    't

     say

     "

    I

     am

     [

    Your

     Name

    ]

    ".

    ]


    Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ],

     and

     I

    'm

     a

     [

    insert

     your

     profession

    ].

     I

    'm

     a

     [

    insert

     your

     occupation

    ]

     and

     I

    've

     always

     been

     passionate

     about

     [

    insert

     something

     you

     like

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     capital

     of

     France

    .

     It

     is

     located

     on

     the

     River

     Se

    ine

    ,

     in

     the

     suburbs

     of

     Paris

    ,

     and

     is

     the

     seat

     of

     the

     Government

     of

     France

    .

     The

     city

     has

     been

     a

     major

     European

     center

     for

     a

     long

     time

    ,

     and

     it

     is

     famous

     for

     its

     art

    ,

     architecture

    ,

     food

    ,

     fashion

    ,

     and

     music

    .

     It

     is

     home

     to

     many

     famous

     landmarks

    ,

     including

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

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     the

     home

     of

     the

     French

     Academy

    ,

     which

     was

     established

     in

     

    1

    7

    9

    3

     as

     the

     first

     permanent

     academy

     of

     arts

     in

     Europe

    .

     Despite

     being

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     significant

     growth

    ,

     innovation

    ,

     and

     integration

     with

     new

     technologies

    .

     Here

     are

     some

     potential

     trends

     to

     consider

    :
    


    1

    .

     Increased

     accuracy

     and

     reliability

    :

     AI

     will

     continue

     to

     improve

     in

     its

     ability

     to

     recognize

     patterns

     and

     make

     accurate

     predictions

    .

     This

     will

     lead

     to

     more

     precise

     and

     reliable

     applications

    ,

     such

     as

     natural

     language

     processing

    ,

     image

     recognition

    ,

     and

     autonomous

     vehicles

    .
    


    2

    .

     Autonomous

     decision

    -making

    :

     AI

     will

     become

     more

     capable

     of

     making

     decisions

     based

     on

     data

     and

     context

    ,

     without

     human

     intervention

    .

     This

     will

     allow

     for

     more

     efficient

    ,

     effective

    ,

     and

     risk

    -

    averse

     decision

    -making

    .
    


    3

    .

     Increased

     transparency

    :

     AI

     will

     become

     more

     transparent

     and

     accessible

    ,

     allowing

     for

     greater

     public

     understanding

    



```python
llm.shutdown()
```
