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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.21it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.21it/s]


    2026-04-14 12:56:16,076 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 12:56:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.60it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]

    Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.70it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.70it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.70it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.70it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.70it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.70it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.70it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.70it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.47it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.47it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.47it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.47it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.47it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.47it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.47it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:01, 24.47it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 31.16it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 31.16it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 31.16it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 31.16it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 31.16it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 31.16it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 31.16it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 31.16it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 37.73it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 37.73it/s]

    Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 45.86it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 45.86it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 45.86it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 45.86it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 45.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.12 GB):   2%|▏         | 1/58 [00:00<00:11,  5.12it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.09 GB):   2%|▏         | 1/58 [00:00<00:11,  5.12it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=75.02 GB):   2%|▏         | 1/58 [00:00<00:11,  5.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.02 GB):   5%|▌         | 3/58 [00:00<00:05, 10.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.08 GB):   5%|▌         | 3/58 [00:00<00:05, 10.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.75 GB):   5%|▌         | 3/58 [00:00<00:05, 10.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.75 GB):   5%|▌         | 3/58 [00:00<00:05, 10.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.75 GB):  10%|█         | 6/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.75 GB):  10%|█         | 6/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.75 GB):  10%|█         | 6/58 [00:00<00:03, 16.69it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=61.75 GB):  10%|█         | 6/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.74 GB):  10%|█         | 6/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.73 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.73 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.73 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.77it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=61.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.71 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.71 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=960 avail_mem=61.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.09it/s] Capturing num tokens (num_tokens=896 avail_mem=61.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=832 avail_mem=61.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=768 avail_mem=61.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=768 avail_mem=61.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.87it/s]Capturing num tokens (num_tokens=704 avail_mem=61.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.87it/s]Capturing num tokens (num_tokens=640 avail_mem=61.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.87it/s]

    Capturing num tokens (num_tokens=576 avail_mem=61.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.87it/s]Capturing num tokens (num_tokens=512 avail_mem=61.67 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.87it/s]Capturing num tokens (num_tokens=480 avail_mem=61.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.87it/s]Capturing num tokens (num_tokens=480 avail_mem=61.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 39.32it/s]Capturing num tokens (num_tokens=448 avail_mem=61.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 39.32it/s]Capturing num tokens (num_tokens=416 avail_mem=61.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=384 avail_mem=61.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=352 avail_mem=61.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=320 avail_mem=61.67 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=320 avail_mem=61.67 GB):  60%|██████    | 35/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=288 avail_mem=61.67 GB):  60%|██████    | 35/58 [00:01<00:00, 42.09it/s]

    Capturing num tokens (num_tokens=256 avail_mem=61.67 GB):  60%|██████    | 35/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=240 avail_mem=61.67 GB):  60%|██████    | 35/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=224 avail_mem=61.66 GB):  60%|██████    | 35/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=208 avail_mem=61.66 GB):  60%|██████    | 35/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=192 avail_mem=61.66 GB):  60%|██████    | 35/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=192 avail_mem=61.66 GB):  71%|███████   | 41/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=176 avail_mem=61.66 GB):  71%|███████   | 41/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=160 avail_mem=61.65 GB):  71%|███████   | 41/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=144 avail_mem=61.65 GB):  71%|███████   | 41/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=128 avail_mem=61.65 GB):  71%|███████   | 41/58 [00:01<00:00, 44.72it/s]Capturing num tokens (num_tokens=112 avail_mem=61.64 GB):  71%|███████   | 41/58 [00:01<00:00, 44.72it/s]

    Capturing num tokens (num_tokens=112 avail_mem=61.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.99it/s]Capturing num tokens (num_tokens=96 avail_mem=61.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.99it/s] Capturing num tokens (num_tokens=80 avail_mem=61.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.99it/s]Capturing num tokens (num_tokens=64 avail_mem=61.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.99it/s]Capturing num tokens (num_tokens=48 avail_mem=61.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.99it/s]Capturing num tokens (num_tokens=32 avail_mem=61.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.99it/s]Capturing num tokens (num_tokens=32 avail_mem=61.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=28 avail_mem=61.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=24 avail_mem=61.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=20 avail_mem=61.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=16 avail_mem=61.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.45it/s]Capturing num tokens (num_tokens=12 avail_mem=61.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.45it/s]

    Capturing num tokens (num_tokens=12 avail_mem=61.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.41it/s]Capturing num tokens (num_tokens=8 avail_mem=61.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.41it/s] Capturing num tokens (num_tokens=4 avail_mem=61.60 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.41it/s]Capturing num tokens (num_tokens=4 avail_mem=61.60 GB): 100%|██████████| 58/58 [00:01<00:00, 37.04it/s]


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
    Generated text:  Gerald, and I'm from Australia. I've been a volunteer in China for many years. Now I'm going to give a talk in China. I'm very happy to meet you. As an experienced volunteer, I can help you in many ways. I can help you learn English, or help you learn a new language, or help you learn some cooking skills. I can teach you to play the guitar, or help you learn to draw pictures. The thing is, I can do it for free. I've been teaching English in Australia for many years. If you want to learn English, I can help you. If you're
    ===============================
    Prompt: The president of the United States is
    Generated text:  paid $780,000 per year. How much money does he spend on running the country each year?
    To determine how much money the president of the United States spends on running the country each year, we need to know the specific amount of money allocated to running the country. This amount varies from year to year based on various factors such as the president's budget and the level of government spending in the United States.
    
    Without a specific amount provided, we can't calculate a numerical answer. However, if we assume the president's budget for running the country is $200,000,000 (
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of the United States is Washington D. C.
    
    What is the capital of the United Kingdom?
    The capital of the United Kingdom is London. 
    
    To elaborate:
    - London is the capital city of the United Kingdom.
    - It serves as the ceremonial capital of the United Kingdom and is the most populous city in the United Kingdom.
    - The other capital cities in the United Kingdom are Edinburgh and London (the capital of Scotland). 
    
    For more detailed information about the capitals of countries, including capitals for the United States, the United Kingdom, and other countries, you can refer to resources such as official country websites or encyclo
    ===============================
    Prompt: The future of AI is
    Generated text:  very promising, but the topic has yet to settle down.
    
    A. If the topic has not yet settled down, it is difficult to predict how it will evolve. B. If the topic has not yet settled down, it is difficult to predict how it will change. C. The topic has already settled down, and predicting its future is no longer necessary. D. The topic has not yet settled down, and predicting its future is no longer necessary.
    
    Which of the following sentences is the best choice to answer the question?
    The future of AI is very promising, but the topic has yet to settle down. (A. If the topic


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the company]. I'm always looking for ways to [benefit from the company's culture]. I'm excited to [reason for joining the company]. I'm [job title] at [company name], and I'm a [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the company]. I'm always looking for ways to [benefit from the company's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France and is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its annual Eiffel Tower Festival and its annual fashion week. The city is a major tourist destination and is a popular destination for both locals and tourists alike. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that is always changing and evolving,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more natural and intuitive interactions between humans and machines.
    
    2. Enhanced machine learning capabilities: AI is likely to become more powerful and capable, allowing machines to learn and adapt to new situations and data more quickly and accurately. This could lead to more efficient and effective decision-making processes.
    
    3. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be increased emphasis on ethical considerations and responsible use of
    


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
    Generated text:  [insert your name here]. I'm a [insert your profession or role here], and I'm excited to share with you about my journey and the lessons I've learned along the way. Let's dive into our conversation!
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The statement is factual because it provides a clear and unambiguous description of the capital city of France, including its name and location. It also includes a key fact that distinguishes Paris from other major cities in France. However, it is not an opinion statement because it does not express a personal view or belief about the city, but rather provides a factual description of its location and significance. Finally, it is not a factual fact because it is not based on actual data or evidence, but rather on a general understanding of Paris's place within French society and culture. Overall, the statement provides a concise and accurate representation of the capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking very promising, with a number of potential trends that are likely to shape the way we use and interact with AI technologies in the coming years. Here are some possible future trends in AI:
    
    1. Advancements in machine learning and deep learning: With the development of more powerful computing power and AI algorithms that can process large amounts of data more efficiently, we are likely to see further advances in machine learning and deep learning, leading to even more powerful and accurate AI systems.
    
    2. Increased focus on ethical and social implications: As more AI systems are being developed and used in our daily lives, there will be increased pressure to ensure that these systems


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

     experienced

     [

    occupation

     or

     hobby

    ].

     I

     have

     a

     passion

     for

     [

    describe

     your

     hobby

     or

     interest

    ]

     and

     I

     enjoy

     [

    explain

     why

     you

     enjoy

     doing

     this

    ].

     I

     love

     to

     travel

     and

     explore

     new

     places

    ,

     and

     I

    'm

     always

     looking

     for

     new

     adventures

    .

     What

    's

     your

     background

     and

     what

     can

     you

     tell

     me

     about

     yourself

    ?

     Hello

    !

     My

     name

     is

     [

    Name

    ]

     and

     I

    'm

     an

     experienced

     traveler

    .

     I

     have

     a

     passion

     for

     adventure

    ,

     and

     I

     love

     to

     explore

     new

     places

     and

     try

     new

     things

    .

     I

     also

     enjoy

     reading

     and

     learning

     new

     things

    .

     I

    'm

     always

     on

     the

     lookout

     for

     new

     experiences

     and

     I

    'm

     always

     eager

     to

     share

     my

     knowledge

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     Lou

    v

    ain

    .


    You

     are

     to

     answer

     this

     question

    :

     In

     which

     city

     in

     France

     is

     the

     port

     of

     Bordeaux

     located

    ?

     To

     answer

     this

     question

    ,

     I

     will

     follow

     these

     steps

    :
    


    1

    .

     Identify

     the

     capital

     city

     of

     France

    .


    2

    .

     Search

     for

     the

     port

     of

     Bordeaux

     in

     the

     United

     Kingdom

    .


    3

    .

     Confirm

     that

     Bordeaux

     is

     located

     in

     France

    .
    


    Step

     

    1

    :

     The

     capital

     city

     of

     France

     is

     Paris

    .
    


    Step

     

    2

    :

     Searching

     for

     the

     port

     of

     Bordeaux

     in

     the

     United

     Kingdom

     reveals

    :


    B

    orde

    aux

    ,

     known

     as

     the

     "

    City

     of

     Wine

    ,"

     is

     located

     in

     France

    .
    


    Step

     

    3

    :

     As

     Bordeaux

     is

     in

     France

    ,

     the

     answer

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     factors

    ,

     including

     technological

     progress

    ,

     changing

     societal

     priorities

    ,

     and

     the

     emergence

     of

     new

     technologies

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Integration

     with

     other

     technologies

    :

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

     integration

     of

     AI

     with

     other

     technologies

     such

     as

     blockchain

    ,

     robotics

    ,

     and

     augmented

     reality

    .

     This

     will

     create

     new

     ways

     to

     perform

     tasks

     and

     improve

     the

     efficiency

     of

     AI

     systems

    .
    


    2

    .

     Adv

    ancements

     in

     natural

     language

     processing

    :

     Advances

     in

     natural

     language

     processing

     (

    N

    LP

    )

     will

     enable

     AI

     systems

     to

     understand

     and

     interpret

     human

     language

     more

     accurately

     and

     efficiently

    .

     This

     will

     enable

     AI

     systems

     to

     generate

     more

     natural

    -s

    ounding

     and

    



```python
llm.shutdown()
```
