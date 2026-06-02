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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.82it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.02it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.02it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.02it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.02it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.02it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.02it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.02it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:04,  8.02it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:04,  8.02it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 20.05it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:05<00:00, 29.30it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 40.80it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 40.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.30it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.22it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.22it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.28it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.11it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.11it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.11it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.11it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.11it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.11it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  71%|███████   | 41/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 46.60it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 46.60it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.10it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.10it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.10it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.10it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.10it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.10it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.95it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.97it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 42.12it/s]


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
    Generated text:  Sam. I am a student at Harvard University and I like to play football. One day I was walking along the street when I saw two football players, looking very nervous. I asked them what was wrong and they told me that they were about to lose the game. It was difficult to believe them, but they had practiced their football for years. They were really good at it. They were the only ones in the game who didn't have any injuries. After that, I was very surprised. The next day, I asked the players what was wrong. "No problem, " they said. "We are just nervous. We don
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 3 inches tall. If it's a sunny day, the average person's height is approximately 70% taller than the president's height. However, if it's a rainy day, the average person's height is 100% taller than the president's height. What is the height difference between a person who is on a sunny day and a person who is on a rainy day?
    
    To find the height difference between a person who is on a sunny day and a person who is on a rainy day, we need to follow these steps:
    
    1. Determine the height of a person who is on a sunny
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Rome
    D. Madrid
    
    A. Paris. Paris is the capital of France. It is the largest and most populous city in the country and a major cultural, economic, and political center. The other options, London and Rome, are located in Europe and Madrid is a city in Spain.
    ===============================
    Prompt: The future of AI is
    Generated text:  in the cloud
    The cloud, or the internet of things, is the next big thing in computing. That's right, the cloud is the place where your photos, phone calls, and smart TVs all live. But why would you want to put all your stuff on the cloud? It's pretty simple: You don't need to worry about your data being on a server far away from you. Instead, you can store your stuff on your own computers and devices, and send it back and forth to the cloud as needed.
    There's a few reasons the cloud is becoming more popular.
    The first is that it's cost-effective. The


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is also the seat of the French government and the country's cultural and political center. Paris is a bustling metropolis with a diverse population, and it is home to many famous landmarks such as the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique and fascinating city. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we are likely to see an increase in automation and artificial intelligence in various industries. This could lead to the creation of more efficient and cost-effective solutions, as well as the creation of new jobs that are not currently being filled by humans.
    
    2. Improved privacy and security: As AI technology becomes more advanced, there will be an increased need for privacy
    


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
    Generated text:  [Name], and I am a [role]. I have always been passionate about [one or more] activities or pursuits, and I am determined to achieve [a specific goal] in my life. I am confident and ambitious, and I am always working towards my dreams. I am also humble and willing to learn from my mistakes, and I believe in the power of hard work and perseverance to achieve my goals. I am excited to meet you and share my story with you. [Name] *reflects a warm, optimistic, and approachable personality. The introduction should be concise and informative, highlighting the character's interests, goals,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historical and cultural center that is known for its landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. Paris is also known for its cuisine and fashion. It is home to many international organizations such as the French Embassy and the European Parliament. The city is also famous for its annual festivals such as the Carnival and the La Jolie France. Paris is a vibrant and dynamic city with a diverse population and a rich history. As of 2021, it has a population of over 1. 3 million people. It is a popular tourist destination and a cultural center
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see significant advancements in several key areas, each with its own potential implications. Here are some possible future trends in artificial intelligence:
    
    1. Increased efficiency and productivity: AI is likely to become more prevalent in industries that rely heavily on automation and process optimization. This could lead to increased efficiency and productivity in various sectors, such as manufacturing, logistics, and healthcare.
    
    2. Enhanced human-cyber security: AI will continue to play a crucial role in security measures, providing real-time threat detection, vulnerability assessment, and incident response capabilities. This could lead to more secure systems and networks, as well as better protection for individuals and organizations.
    
    


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

    .

     I

     am

     a

     versatile

     and

     accomplished

     writer

     who

     has

     always

     loved

     to

     explore

     the

     depths

     of

     the

     human

     mind

    .

     I

     have

     a

     natural

     ability

     to

     convey

     complex

     ideas

     with

     clarity

     and

     precision

    ,

     and

     I

     am

     always

     looking

     for

     new

     ways

     to

     share

     my

     thoughts

     and

     experiences

     with

     the

     world

    .

     Whether

     I

     am

     working

     on

     a

     novel

    ,

     a

     blog

     post

    ,

     or

     even

     a

     casual

     conversation

    ,

     I

     strive

     to

     make

     my

     writing

     accessible

     and

     engaging

     for

     readers

     of

     all

     backgrounds

    .

     I

     am

     a

     professional

     writer

     and

     I

     believe

     that

     the

     key

     to

     success

     is

     to

     always

     strive

     for

     excellence

    ,

     even

     if

     it

     means

     taking

     risks

     and

     trying

     new

     things

    .

     What

    's

     your

     favorite

     hobby

     or

     activity

     to

     do

     when

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

    ,

     the

     world

    -f

    amous

     French

     capital

    ,

     is

     a

     cosm

    opolitan

     met

    ropolis

     with

     a

     rich

     history

    ,

     known

     for

     its

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     famous

     for

     its

     diverse

     cuisine

    ,

     fashion

     scene

    ,

     and

     world

    -class

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     It

     is

     also

     a

     major

     hub

     for

     culture

    ,

     tourism

    ,

     and

     political

     activities

    ,

     attracting

     millions

     of

     visitors

     annually

    .

     Paris

     is

     a

     cultural

     and

     economic

     powerhouse

     that

     plays

     a

     significant

     role

     in

     French

     society

     and

     the

     world

    .

     Given

     its

     status

     as

     a

     global

     capital

    ,

     Paris

     is

     widely

     regarded

     as

     one

     of

     the

     most

     important

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     advancements

     and

     technological

     breakthrough

    s

    .

     Some

     potential

     future

     trends

     include

    :
    


    1

    .

     Increased

     integration

     of

     AI

     with

     other

     technologies

     such

     as

     machine

     learning

    ,

     blockchain

    ,

     and

     quantum

     computing

    .
    


    2

    .

     AI

     becoming

     more

     autonomous

     and

     self

    -aware

    ,

     potentially

     leading

     to

     new

     ethical

     and

     social

     challenges

    .
    


    3

    .

     AI

     systems

     becoming

     more

     capable

     of

     handling

     complex

     problems

     and

     making

     decisions

     that

     have

     long

    -lasting

     impacts

    .
    


    4

    .

     AI

     becoming

     more

     accessible

     and

     affordable

     to

     a

     wider

     range

     of

     people

    ,

     leading

     to

     increased

     adoption

     and

     integration

     in

     various

     industries

    .
    


    5

    .

     AI

     systems

     becoming

     more

     capable

     of

     emotional

     intelligence

     and

     empathy

    ,

     potentially

     leading

     to

     more

     compassionate

     and

     socially

     responsible

     AI

    .
    


    6

    .

     AI

    



```python
llm.shutdown()
```
