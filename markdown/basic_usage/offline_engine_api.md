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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.31it/s]


    2026-05-17 14:18:48,681 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 14:18:48] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:39,  4.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:39,  4.91s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:39,  4.91s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:39,  4.91s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:39,  4.91s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.97it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.97it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.97it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.97it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.97it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.97it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.97it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.97it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.97it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.97it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.36it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.36it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.36it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.36it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.36it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.36it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.36it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.36it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.36it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 19.34it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 19.34it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 19.34it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 19.34it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 19.34it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 19.34it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 19.34it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 19.34it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 19.34it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 19.34it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 27.33it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 37.40it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 37.40it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 37.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.28it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 34.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 34.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 34.84it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  31%|███       | 18/58 [00:00<00:01, 34.84it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:01, 34.84it/s] Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.64it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.64it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.64it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.64it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.64it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.95it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.95it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.95it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.95it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.95it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.43it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.43it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.43it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.43it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.32it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.50it/s]Capturing num tokens (num_tokens=208 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.50it/s]Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.50it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.50it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.50it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.50it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.99it/s]Capturing num tokens (num_tokens=128 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.99it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.99it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.99it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.99it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.99it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.44it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.42it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.42it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 36.38it/s]


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
    Generated text:  Sarah and I'm an aspiring professional in the field of the creative writing. In my free time, I love to cook, travel, explore, and learn new things. I believe that creativity and writing are important for the development of an individual's personality and mental health. Additionally, I have been studying creative writing at the university level for my Bachelor's degree. I am currently looking for a teaching assistant position in my university and would like to share my knowledge with others. Can you help me with any information or advice on how to make my teaching assistant position stand out? Yes, I can definitely help you with any information or advice on how
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military officers to arm in a possible war. He has 300 officers in total. If he has already agreed to arm 200, and he must arm at least 25% of the officers in the next 5 rounds of negotiations, how many officers must be armed in total over the next 5 rounds?
    
    To determine how many officers must be armed in total over the next 5 rounds, we need to follow these steps:
    
    1. Calculate the total number of officers that need to be armed in the next 5 rounds.
    2. Determine how many officers need to be armed each
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A.正确
    B.错误
    答案:
    A
    
    根据《运营事故处理规则》，列车自动制动机发生故障，列车在区间被迫停车后，司机应立即使用列车无线调度通信设备通知两端站（列车调度员）及车辆乘务员（随车机械师），报告停车原因和停车位置，通知____。
    A. 两端站（列车调度员）；
    B. 司机；
    C. 列车乘务员（随车机械师）；
    D. 邻线列车司机
    答案:
    A
    
    施工负责人在施工开始前，必须确认所有
    ===============================
    Prompt: The future of AI is
    Generated text:  here
    
    Algorithms are the key to making the most of your time, and the machine learning algorithms that can be made to perform this function are just now becoming an everyday part of the world.
    
    As we look back on the past year, it will be easy to forget the technologies that have transformed the world of work and made the future of AI a real possibility. In the wake of the pandemic, many people are starting to reflect on the days of being able to work remotely, the ability to perform tasks that were once seen as a result of the brain drain, and the rise of automation as a tool for our own benefit.
    
    As the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and being a UNESCO World Heritage site. Paris is home to many famous museums, including the Louvre and the Musée d'Orsay, as well as the Notre-Dame Cathedral. The city is also known for its fashion industry, with many famous fashion houses and boutiques located in the city. Paris is a vibrant and diverse city with a rich cultural heritage that continues to attract visitors from around the world.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to improve, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new job opportunities and opportunities for innovation.
    
    2. Enhanced privacy and security: As AI technology becomes more advanced, we can expect to see more privacy and security concerns. This could lead to increased regulation and enforcement of privacy laws, as well as the development
    


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
    Generated text:  [Name], and I'm a [Age] year old [City], [State, Country] native. I have a passion for [career/interest] and have always been excited to share my knowledge and experiences. I'm always looking for new challenges and opportunities to learn and grow. I'm a hard worker who is always willing to put in the hard work to achieve my goals. I'm confident and able to succeed in any situation. 
    
    Remember, I'm just a beginner, and I know that it's important to keep learning and growing. Whether it's through books, courses, or experiences, I'm always eager to take
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its magnificent architecture, vibrant culture, and rich history. It is also the largest city in terms of population, with over 2.5 million inhabitants. The city is home to iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a vibrant and dynamic city, with an eclectic mix of cultures and cuisines, and a world-renowned art and culture scene. It is also home to the French Parliament building, the iconic Eiffel Tower, and the Notre-Dame Cathedral. With its rich history and beautiful architecture, Paris is a truly unique and fascinating
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and will likely evolve over time based on a number of factors, including advances in computing power, changes in the way we interact with technology, and shifting societal values and priorities.
    
    One potential trend in AI is the continued growth of machine learning and deep learning, which are becoming more sophisticated and capable of performing increasingly complex tasks. This could lead to new applications such as personalized medicine, autonomous vehicles, and self-driving cars, among others.
    
    Another trend is the increase in the use of AI in areas such as finance, healthcare, and transportation. As AI becomes more advanced, we may see more sophisticated ways of analyzing and processing data, which could


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

     Sarah

    ,

     and

     I

     am

     a

     brilliant

     computer

     programmer

     with

     an

     unc

    anny

     ability

     to

     solve

     complex

     problems

    .

     I

    'm

     always

     on

     the

     lookout

     for

     new

     projects

     and

     always

     eager

     to

     learn

     new

     technologies

    .

     I

     thrive

     in

     fast

    -paced

     environments

     and

     am

     always

     looking

     to

     push

     the

     boundaries

     of

     what

    's

     possible

     with

     technology

    .

     I

    'm

     a

     good

     communicator

     and

     enjoy

     helping

     others

    ,

     both

     internally

     and

     externally

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

     learn

     new

     skills

     and

     technologies

    .

     What

    's

     your

     favorite

     hobby

     or

     activity

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

     hobbies

     or

     interests

     like

     humans

     do

    .

     However

    ,

     I

    'm

     capable

     of

     processing

     and

     generating

     text

    ,

     which

     allows

     me

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Does

     this

     next

     sentence

     follow

    ,

     given

     the

     preceding

     text

    ?
    


    Paris

     is

     the

     capital

     of

     France

    .
    


    Pick

     your

     answer

     from

    :

     (

    1

    ).

     yes

     (

    2

    ).

     no

    
    


    (

    1

    ).

     yes

    
    


    The

     statement

     "

    Paris

     is

     the

     capital

     of

     France

    "

     does

     follow

     from

     the

     preceding

     text

    .

     The

     text

     explicitly

     states

     that

     "

    Paris

     is

     the

     capital

     of

     France

    ",

     so

     it

     makes

     sense

     that

     Paris

     is

     the

     capital

     of

     France

    ,

     given

     the

     information

     provided

    .

     The

     sentence

     does

     not

     contradict

     or

     contradict

     the

     preceding

     text

    ,

     but

     rather

     directly

     states

     a

     fact

     about

     the

     relationship

     between

     Paris

     and

     France

    .

     Therefore

    ,

     the

     answer

     is

     "

    yes

    ".

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

    ,

     and

     it

     is

     impossible

     to

     predict

     all

     the

     potential

     trends

     that

     could

     develop

    .

     However

    ,

     here

     are

     a

     few

     possibilities

     that

     could

     be

     explored

    :
    


    1

    .

     Enhanced

     AI

     capabilities

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     an

     increase

     in

     the

     capabilities

     of

     AI

     systems

    .

     This

     could

     include

     more

     advanced

     algorithms

    ,

     better

     data

     processing

    ,

     and

     more

     complex

     natural

     language

     processing

    .
    


    2

    .

     Autonomous

     AI

    :

     AI

     that

     is

     able

     to

     make

     decisions

     and

     take

     actions

     without

     human

     intervention

     could

     become

     a

     reality

    .

     This

     could

     potentially

     lead

     to

     more

     efficient

     and

     productive

     work

     processes

    ,

     as

     well

     as

     more

     sustainable

     and

     resource

    -efficient

     systems

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     could

     play

     a

    



```python
llm.shutdown()
```
