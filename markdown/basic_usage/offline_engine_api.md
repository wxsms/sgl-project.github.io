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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.69it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:31,  1.66it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 23.12it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 23.12it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 23.12it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 23.12it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 23.12it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 23.12it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 23.12it/s]

    Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 23.12it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 23.12it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 23.12it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 31.31it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 31.31it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 31.31it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 31.31it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 31.31it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 31.31it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:04<00:00, 31.31it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:04<00:00, 31.31it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:04<00:00, 31.31it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:04<00:00, 31.31it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:04<00:00, 31.31it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:04<00:00, 41.27it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:04<00:00, 41.27it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:04<00:00, 41.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.95 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.95 GB):   2%|▏         | 1/58 [00:00<00:06,  8.19it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.92 GB):   2%|▏         | 1/58 [00:00<00:06,  8.19it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=38.93 GB):   2%|▏         | 1/58 [00:00<00:06,  8.19it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.93 GB):   5%|▌         | 3/58 [00:00<00:04, 11.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.93 GB):   5%|▌         | 3/58 [00:00<00:04, 11.91it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.93 GB):   5%|▌         | 3/58 [00:00<00:04, 11.91it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.93 GB):   5%|▌         | 3/58 [00:00<00:04, 11.91it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.93 GB):  10%|█         | 6/58 [00:00<00:02, 17.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.92 GB):  10%|█         | 6/58 [00:00<00:02, 17.76it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.91 GB):  10%|█         | 6/58 [00:00<00:02, 17.76it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=38.91 GB):  10%|█         | 6/58 [00:00<00:02, 17.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.91 GB):  10%|█         | 6/58 [00:00<00:02, 17.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.91 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.90 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.90 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.90 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.99it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.90 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.89 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.89 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.89 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.76it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.89 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.76it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=38.88 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.76it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.88 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.88 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.88 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.86 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.09it/s]Capturing num tokens (num_tokens=960 avail_mem=38.87 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.09it/s] Capturing num tokens (num_tokens=896 avail_mem=38.87 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.09it/s]Capturing num tokens (num_tokens=832 avail_mem=38.87 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.09it/s]Capturing num tokens (num_tokens=768 avail_mem=38.86 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.09it/s]Capturing num tokens (num_tokens=768 avail_mem=38.86 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.97it/s]Capturing num tokens (num_tokens=704 avail_mem=38.86 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.97it/s]Capturing num tokens (num_tokens=640 avail_mem=38.86 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.97it/s]

    Capturing num tokens (num_tokens=576 avail_mem=38.86 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.97it/s]Capturing num tokens (num_tokens=512 avail_mem=38.84 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.97it/s]Capturing num tokens (num_tokens=480 avail_mem=38.86 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.97it/s]Capturing num tokens (num_tokens=480 avail_mem=38.86 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.44it/s]Capturing num tokens (num_tokens=448 avail_mem=38.85 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.44it/s]Capturing num tokens (num_tokens=416 avail_mem=38.85 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.44it/s]Capturing num tokens (num_tokens=384 avail_mem=38.85 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.44it/s]Capturing num tokens (num_tokens=352 avail_mem=38.84 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.44it/s]Capturing num tokens (num_tokens=320 avail_mem=38.84 GB):  52%|█████▏    | 30/58 [00:01<00:00, 42.44it/s]Capturing num tokens (num_tokens=320 avail_mem=38.84 GB):  60%|██████    | 35/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=288 avail_mem=38.66 GB):  60%|██████    | 35/58 [00:01<00:00, 43.79it/s]

    Capturing num tokens (num_tokens=256 avail_mem=37.73 GB):  60%|██████    | 35/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=240 avail_mem=37.73 GB):  60%|██████    | 35/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=224 avail_mem=37.73 GB):  60%|██████    | 35/58 [00:01<00:00, 43.79it/s]

    Capturing num tokens (num_tokens=208 avail_mem=37.67 GB):  60%|██████    | 35/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=208 avail_mem=37.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=192 avail_mem=37.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=176 avail_mem=37.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=160 avail_mem=37.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=144 avail_mem=37.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=128 avail_mem=37.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=128 avail_mem=37.66 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=112 avail_mem=37.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=96 avail_mem=37.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.44it/s] Capturing num tokens (num_tokens=80 avail_mem=37.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.44it/s]

    Capturing num tokens (num_tokens=64 avail_mem=37.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=48 avail_mem=37.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=48 avail_mem=37.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=32 avail_mem=37.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=28 avail_mem=37.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=24 avail_mem=37.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=20 avail_mem=37.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=16 avail_mem=37.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=16 avail_mem=37.63 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.52it/s]Capturing num tokens (num_tokens=12 avail_mem=37.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.52it/s]Capturing num tokens (num_tokens=8 avail_mem=37.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.52it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=37.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.52it/s]Capturing num tokens (num_tokens=4 avail_mem=37.61 GB): 100%|██████████| 58/58 [00:01<00:00, 32.81it/s]


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
    Generated text:  Robin. I'm a college student at University of Wisconsin-Madison. My parents are both teachers in a school. My father works with children. My mother works with adults. I grew up in an apartment. The apartment has three bedrooms and my parents and I stay in the house. My parents are both very busy. My parents work very late into the night and sometimes they have to take care of my sister and me. My father works very early in the morning and then he will spend a lot of time with his daughter. My mother works very early in the morning and then she will spend a lot of time with her daughter.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the Cabinet of the executive branch of the government. The President of the United States is the leader of the United States federal government, and is responsible for making and enforcing laws. There are currently 44 cabinet members of the executive branch of the United States government. However, the cabinet members are not in charge of the executive branch directly. They are appointed by the President of the United States, and the President appoints the other members of the cabinet. The Cabinet is not a title of the President, but a description of the positions within the Cabinet that are appointed by the President.
    Does this next sentence follow, given the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Tokyo
    D. Moscow
    
    To determine the capital of France, we can follow these steps:
    
    1. **Identify the capital of France**: France is a country, so its capital is typically the country's name. Therefore, the capital of France is not a specific city but the name of the country.
    
    2. **Consider the given options**: The options provided are Paris, London, Tokyo, and Moscow. Since the capital of France is not a specific city but the name of the country, the correct answer is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the developers and creators. Whether you're a professional or amateur, there are many ways to advance your knowledge and skills in the field. One of the most exciting areas of AI is machine learning, which involves the creation of algorithms that can learn and improve based on data. With the right tools and techniques, you can use machine learning to build models that can be used to make predictions and recommendations, without ever having to provide explicit instructions.
    Machine learning is one of the most promising areas of AI, and it has the potential to change the way we live and work. By using machine learning to analyze and process data, we


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


    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its annual festivals and events, including the Eiffel Tower Parade and the World Cup. The city is a major economic and cultural center in France and plays a significant role in the country's political and social life. Paris is a popular tourist destination
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into various industries, from manufacturing to healthcare to transportation. This will lead to increased automation of tasks, which will require more human workers to perform the tasks that are now being done by machines.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be a growing concern about the ethical implications of AI. This will include issues such as bias, transparency, and accountability. There
    


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
    Generated text:  [Your Name], and I'm a [Career Goal] who has been [X] for [X] years. I'm confident, professional, and always up for a challenge. I love [Your Hobby or Passion], and I'm always looking for ways to expand my skills and knowledge. I'm eager to learn and grow, and I'm always willing to share my knowledge with others. I'm a true mentor and someone who always strives to be a good role model. I'm ready to connect with anyone who is looking for a similar kind of person. Thank you for considering me for a job. [Your Name] [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is known for its iconic landmarks like the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and other historical buildings. It is a major center for culture, art, and international affairs. The city is also known for its fashion and gastronomy, particularly in the Parisian district of Montmartre and the neighborhoods around the Seine River. The French language is the official language, and it is also home to numerous museums, theaters, and other cultural institutions. Paris is a bustling city with a diverse and vibrant culture, and it continues to be a world-renowned capital city. It is often called the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and highly promising, with significant potential to revolutionize the way we live, work, and interact with technology. Here are some of the key trends shaping the future of AI:
    
    1. Increased emphasis on ethics and responsibility: As AI becomes more prevalent in our lives, there will be a growing emphasis on how it should be used and how it should be responsible for its decisions and actions. This includes considerations of bias, fairness, transparency, and accountability.
    
    2. Development of more advanced AI: With ongoing research and development, we can expect to see significant improvements in the capabilities of AI systems. For example, we may see the development of


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

     am

     [

    Age

    ]

     years

     old

    .

     I

     am

     a

     [

    Occup

    ation

     or

     Profession

    ]

     with

     [

    Number

     of

     Years

     in

     Industry

     or

     Role

    ].

     I

     am

     passionate

     about

     [

    Your

     Interest

     or

     Subject

    ],

     and

     I

     strive

     to

     use

     my

     knowledge

     and

     skills

     to

     [

    Your

     Goal

     or

     Objective

    ].

     I

     am

     a

     [

    Your

     Profession

     or

     Expert

    ise

    ]

     with

     [

    Number

     of

     Years

     in

     Industry

     or

     Role

    ].

     I

     am

     always

     looking

     for

     opportunities

     to

     [

    Your

     Goal

     or

     Objective

    ]

     and

     I

     am

     dedicated

     to

     [

    Your

     Interest

     or

     Subject

    ].

     Thank

     you

    .

     


    (P

    lease

     make

     sure

     to

     use

     a

     neutral

     and

     positive

     tone

     throughout

     your

     self

    -int

    roduction

    ,

     and

     to

     keep

     it

     brief

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     The

     city

     is

     a

     bustling

     hub

     of

     culture

    ,

     fashion

    ,

     and

     nightlife

    ,

     with

     its

     landmarks

     serving

     as

     a

     symbol

     of

     France

    ’s

     rich

     history

     and

     cultural

     heritage

    .

     
    


    The

     city

    's

     economy

     is

     based

     on

     tourism

    ,

     with

     its

     famous

     landmarks

     attracting

     millions

     of

     visitors

     each

     year

    ,

     including

     celebrities

    ,

     business

     leaders

    ,

     and

     visitors

     from

     around

     the

     world

    .

     Paris

     is

     also

     home

     to

     many

     important

     universities

     and

     research

     institutions

    ,

     and

     its

     place

     in

     global

     affairs

     makes

     it

     a

     key

     player

     in

     French

     politics

     and

     diplomacy

    .

     
    


    Paris

     is

     known

     for

     its

     modern

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     potential

     trends

     that

     could

     shape

     the

     direction

     of

     AI

     research

     and

     development

    .

     Some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     accuracy

     and

     efficiency

    :

     AI

     is

     expected

     to

     continue

     improving

     its

     ability

     to

     perform

     tasks

     faster

     and

     more

     accurately

     than

     current

     human

    -made

     systems

    .

     This

     will

     lead

     to

     more

     efficient

     and

     cost

    -effective

     applications

     of

     AI

     technology

    .
    


    2

    .

     Personal

    ization

     and

     adapt

    ability

    :

     AI

     is

     becoming

     increasingly

     capable

     of

     understanding

     and

     adapting

     to

     the

     needs

     and

     preferences

     of

     individuals

    ,

     leading

     to

     more

     personalized

     and

     adaptable

     systems

     that

     can

     learn

     and

     improve

     over

     time

    .
    


    3

    .

     Autonomous

     and

     self

    -driving

     vehicles

    :

     AI

     is

     already

     being

     used

     in

     vehicles

    



```python
llm.shutdown()
```
