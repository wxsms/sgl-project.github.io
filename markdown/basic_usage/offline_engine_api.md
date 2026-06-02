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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.59it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.42it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.90it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.02it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 33.01it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 33.01it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.01it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.01it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.01it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.01it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.01it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.36 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.36 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.36 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.36 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.36 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.36 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.35 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.34 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.34 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.34 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=42.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.32 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.32 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.31 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.31 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.31 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.30 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.30 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=960 avail_mem=39.83 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=37.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=832 avail_mem=37.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=832 avail_mem=37.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.26it/s]Capturing num tokens (num_tokens=768 avail_mem=37.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.26it/s]Capturing num tokens (num_tokens=704 avail_mem=37.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.26it/s]Capturing num tokens (num_tokens=640 avail_mem=37.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.26it/s]Capturing num tokens (num_tokens=576 avail_mem=37.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.26it/s]Capturing num tokens (num_tokens=512 avail_mem=37.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.26it/s]Capturing num tokens (num_tokens=512 avail_mem=37.01 GB):  50%|█████     | 29/58 [00:00<00:00, 41.50it/s]Capturing num tokens (num_tokens=480 avail_mem=37.03 GB):  50%|█████     | 29/58 [00:00<00:00, 41.50it/s]Capturing num tokens (num_tokens=448 avail_mem=37.02 GB):  50%|█████     | 29/58 [00:00<00:00, 41.50it/s]Capturing num tokens (num_tokens=416 avail_mem=37.02 GB):  50%|█████     | 29/58 [00:00<00:00, 41.50it/s]

    Capturing num tokens (num_tokens=384 avail_mem=37.02 GB):  50%|█████     | 29/58 [00:00<00:00, 41.50it/s]Capturing num tokens (num_tokens=352 avail_mem=37.01 GB):  50%|█████     | 29/58 [00:00<00:00, 41.50it/s]Capturing num tokens (num_tokens=352 avail_mem=37.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.61it/s]Capturing num tokens (num_tokens=320 avail_mem=37.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.61it/s]Capturing num tokens (num_tokens=288 avail_mem=37.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.61it/s]Capturing num tokens (num_tokens=256 avail_mem=37.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.61it/s]Capturing num tokens (num_tokens=240 avail_mem=37.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.61it/s]Capturing num tokens (num_tokens=224 avail_mem=37.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.61it/s]Capturing num tokens (num_tokens=224 avail_mem=37.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=208 avail_mem=36.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=192 avail_mem=36.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=176 avail_mem=36.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]

    Capturing num tokens (num_tokens=160 avail_mem=36.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=144 avail_mem=36.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=144 avail_mem=36.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=128 avail_mem=36.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=112 avail_mem=36.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=96 avail_mem=36.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.52it/s] Capturing num tokens (num_tokens=80 avail_mem=36.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=64 avail_mem=36.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.52it/s]Capturing num tokens (num_tokens=64 avail_mem=36.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.62it/s]Capturing num tokens (num_tokens=48 avail_mem=36.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.62it/s]Capturing num tokens (num_tokens=32 avail_mem=36.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.62it/s]Capturing num tokens (num_tokens=28 avail_mem=36.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.62it/s]

    Capturing num tokens (num_tokens=24 avail_mem=36.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.62it/s]Capturing num tokens (num_tokens=20 avail_mem=36.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.62it/s]Capturing num tokens (num_tokens=20 avail_mem=36.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.79it/s]Capturing num tokens (num_tokens=16 avail_mem=36.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.79it/s]Capturing num tokens (num_tokens=12 avail_mem=36.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.79it/s]Capturing num tokens (num_tokens=8 avail_mem=36.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.79it/s] Capturing num tokens (num_tokens=4 avail_mem=36.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.79it/s]Capturing num tokens (num_tokens=4 avail_mem=36.94 GB): 100%|██████████| 58/58 [00:01<00:00, 41.14it/s]


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
    Generated text:  Mark and I am a student at New York University in New York City. My major is Mechanical Engineering, with a minor in chemistry. I am enrolled in a MSc in Biochemical Engineering at the University of Oxford, and currently doing my dissertation research in the laboratory of Prof. Alexo. I have a strong background in electrical engineering and physics, and I also have a background in biology and chemistry. Throughout my undergraduate studies, I have participated in several extracurricular activities, including the New York City Robotics Competition, a robotics competition that tests participants' knowledge and understanding of robotics, and a science fair that tests participants' understanding of
    ===============================
    Prompt: The president of the United States is
    Generated text:  a noble position, a ____ (a/an) ____ ____. (Please choose the correct verb form: position, position, position, position, position, position)
    答案:
    
    position
    
    已知某年全国甲乙两种食品的销售价格分别为25元和20元，其中甲食品的价格比乙食品的价格高20%。请问，甲食品的价格是乙食品价格的多少倍？
    A. 2.5
    B. 3.0
    C. 4.0
    D. 5.0
    答案:
    
    A
    
    下列选项中，哪一项属于行政单位的会计科目
    ===============================
    Prompt: The capital of France is
    Generated text:  ( ). A. Paris B. Lille C. Geneva D. Marseille
    A. The capital of France is ().
    A. Paris
    B. Lille
    C. Geneva
    D. Marseille
    Answer:
    A
    
    The enterprise annuity is a _____ system.
    A. Social Insurance
    B. Public Welfare
    C. Economic Assistance
    D. Social Welfare
    Answer:
    A
    
    When the economic growth rate increases, the national income elasticity of demand for government spending is ( ).
    A. 1
    B. 0
    C. 10
    D. 20
    Answer:
    B
    
    Among the following options
    ===============================
    Prompt: The future of AI is
    Generated text:  very much a high-tech future, and one of the most important aspects of that is the ability to use the many billions of connected devices that now populate our homes, including smart speakers, smart home devices, and smart sensors. These devices are capable of gathering and processing vast amounts of data, and AI is playing a pivotal role in the way that we interpret that data and make sense of it. But in order to achieve true AI, a lot of its building blocks need to be in place. Here are some of the key challenges that are driving the development of these AI systems:
    1. Data Quality
    AI systems are built on the data


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your character]. I enjoy [insert a brief description of your character's interests or hobbies]. I'm [insert a brief description of your character's personality]. I'm always looking for new challenges and opportunities to grow and learn. What do you think makes you unique? I'm [insert a brief description of your character's personality]. I'm always looking for new challenges and opportunities to grow and learn. What do you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its annual fashion and food festivals, and is a popular tourist destination for visitors from around the world. The city is a cultural and economic hub, and is a major center of politics, business, and entertainment in France. It is also home to many important institutions such as the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more natural and intuitive interactions between humans and machines.
    
    2. Enhanced machine learning capabilities: AI is likely to become more powerful and capable of learning from large amounts of data, allowing machines to make more accurate and nuanced predictions and decisions.
    
    3. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations and responsible use of AI. This could lead to more
    


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
    Generated text:  [Name], and I'm an experienced [occupation] with a passion for [a particular skill or hobby]. I enjoy [ways to unwind or relax]. I'm always up for a challenge and looking for new opportunities to grow and learn. If you're looking for someone to help with [some project or task], I'd love to work with you! Let's chat! [Name], what can you tell me about yourself? [Name], what do you enjoy most about your job? [Name], what's your favorite hobby or activity? [Name], what's your biggest challenge? [Name], what's your favorite project to complete
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located on the Seine River in the south of the country. It was founded as a minor town in the 12th century and is now one of the largest cities in the world by population. 
    
    **Paris**, France, known as the City of Light and the "Musee du Louvre", is the seat of government, government, government and is the main city of France. It was founded as a minor town in the 12th century and is now one of the largest cities in the world by population. The city has been home to several important cultural institutions, including the Louvre Museum, the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bound to be rapidly evolving, and there are several potential trends that could significantly shape its development. Here are some potential future trends in AI:
    
    1. Increased Integration with Human Intelligence: As AI continues to become more advanced, it will become increasingly integrated with human intelligence, allowing for better understanding of complex human behavior and decision-making.
    
    2. Personalization and Adaptability: AI will continue to improve its ability to personalize and adapt to users, leading to more efficient and effective user experiences.
    
    3. Ethical and Legal Considerations: As AI becomes more integrated into our daily lives, there will be increased scrutiny and ethical considerations around its development and use


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

    ],

     and

     I

    'm

     a

     [

    job

     or

     hobby

    ]

     with

     a

     passion

     for

     [

    love

    /h

    obby

    ].

     I

     believe

     in

     [

    reason

     why

     you

     believe

     in

     your

     work

    ]

     and

     have

     dedicated

     myself

     to

     [

    career

     goal

     or

     personal

     goal

    ].

     What

     brings

     you

     to

     this

     place

    ?

     [

    What

     motiv

    ates

     you

     to

     go

     there

    ?]

     I

    'm

     always

     looking

     for

     the

     next

     big

     project

     and

     I

    'm

     excited

     to

     apply

     my

     skills

     and

     knowledge

     to

     help

     others

     achieve

     their

     goals

    .

     What

     brings

     you

     to

     this

     place

    ?

     [

    What

     makes

     you

     unique

     in

     your

     field

    ?]

     I

    'm

     always

     open

     to

     learning

     and

     experimenting

     with

     new

     ideas

    .

     I

    'm

     driven

     by

     a

     desire

     to

     make

     a

     positive

     impact

     on

     the

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

      
    


    Rate

     the

     content

     to

     ensure

     that

     it

     encompasses

     a

     broad

     range

     of

     information

     while

     avoiding

     bias

    .

     Yes

    ,

     I

     can

     provide

     a

     concise

     factual

     statement

     about

     the

     capital

     of

     France

    ,

     Paris

    ,

     and

     include

     information

     on

     its

     historical

     significance

    ,

     notable

     landmarks

    ,

     art

     and

     architecture

    ,

     cuisine

    ,

     and

     other

     cultural

     aspects

    .

     Here

    's

     the

     statement

    :
    


    Paris

    ,

     the

     capital

     city

     of

     France

    ,

     is

     renowned

     for

     its

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     rich

     artistic

     and

     cultural

     heritage

    ,

     including

     the

     world

    -f

    amous

     M

    uses

    ,

     the

     Par

    then

    on

    ,

     and

     the

     Op

    éra

    .

     The

     city

    's

     cuisine

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     driven

     by

     several

     key

     trends

    :
    


    1

    .

     Increased

     specialization

     and

     automation

    :

     AI

     is

     increasingly

     being

     used

     to

     automate

     tasks

     that

     were

     previously

     done

     by

     humans

    .

     For

     example

    ,

     AI

     systems

     can

     be

     used

     to

     perform

     routine

     and

     repetitive

     tasks

    ,

     reducing

     the

     need

     for

     manual

     labor

    .

     This

     trend

     is

     expected

     to

     continue

     as

     AI

     technologies

     continue

     to

     improve

     and

     become

     more

     efficient

    .
    


    2

    .

     Increased

     integration

     with

     human

     intelligence

    :

     AI

     is

     expected

     to

     be

     integrated

     with

     human

     intelligence

     in

     new

     and

     exciting

     ways

    .

     For

     example

    ,

     AI

     systems

     can

     be

     designed

     to

     understand

     and

     adapt

     to

     human

     behavior

     patterns

    ,

     such

     as

     using

     machine

     learning

     algorithms

     to

     predict

     and

     adjust

     the

     intensity

     of

     emotions

     in

     conversations

     with

     humans

    



```python
llm.shutdown()
```
