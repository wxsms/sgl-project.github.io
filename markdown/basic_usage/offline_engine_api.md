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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.37it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.47it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.47it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.47it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.47it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.47it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.47it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.47it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.47it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.47it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.85it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.85it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.85it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.85it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.85it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.85it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.85it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.85it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.85it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 14.85it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 22.10it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 22.10it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 22.10it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 22.10it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 22.10it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 22.10it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 22.10it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 22.10it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 22.10it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 22.10it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 30.43it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 30.43it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 30.43it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 30.43it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 30.43it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 30.43it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 30.43it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 30.43it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   2%|▏         | 1/58 [00:00<00:13,  4.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   2%|▏         | 1/58 [00:00<00:13,  4.34it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.03 GB):   2%|▏         | 1/58 [00:00<00:13,  4.34it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.03 GB):   5%|▌         | 3/58 [00:00<00:05, 10.31it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.03 GB):   5%|▌         | 3/58 [00:00<00:05, 10.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   5%|▌         | 3/58 [00:00<00:05, 10.31it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   9%|▊         | 5/58 [00:00<00:04, 12.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:04, 12.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:04, 12.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:04, 12.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.42it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.78it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.78it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.78it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.78it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:01<00:01, 27.54it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:01<00:01, 27.54it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:01<00:01, 27.54it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.67it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.67it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.67it/s]

    Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.67it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.67it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.67it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.31it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.31it/s]

    Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.31it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.31it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.31it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.31it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.09it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.71it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.71it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.71it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.71it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.71it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.71it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.49it/s]

    Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.46it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.46it/s]Capturing num tokens (num_tokens=12 avail_mem=76.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.46it/s]Capturing num tokens (num_tokens=8 avail_mem=76.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.46it/s] Capturing num tokens (num_tokens=4 avail_mem=76.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.46it/s]Capturing num tokens (num_tokens=4 avail_mem=76.59 GB): 100%|██████████| 58/58 [00:01<00:00, 29.87it/s]


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
    Generated text:  Alex and I'm a 25 year old male. I've got a 17 year old girlfriend who is very supportive and understanding. I'm not a very good driver but I don't want to offend her. I just wanted to ask if it's possible to get a passenger seat on a car that is not in your car. I realize that it's a bit awkward if the passenger is in the front, but I would like to get a seat to help make sure we don't squish together. 
    
    I understand that sometimes it's a challenge to get someone in the passenger seat but I was hoping that someone might know
    ===============================
    Prompt: The president of the United States is
    Generated text:  a presidential candidate. The number of votes he gets is 1/3 of the total votes cast. If there are 300 voters, what is the number of votes the candidate gets? To determine the number of votes the presidential candidate gets, we need to follow these steps:
    
    1. Identify the total number of votes cast.
    2. Calculate the number of votes the candidate receives based on the given ratio.
    
    First, we know that the total number of voters is 300. According to the problem, the number of votes the candidate gets is \(\frac{1}{3}\) of the total votes cast.
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. London
    B. Paris
    C. Moscow
    D. Shanghai
    Answer:
    B
    
    The main difference between an open system and a closed system lies in ____.
    A. The pressure difference inside the system
    B. The volume of the system
    C. The heat exchange between the system and its surroundings
    D. The temperature difference between the system and its surroundings
    Answer:
    C
    
    What is the most common cause of chronic renal failure?
    A. Renal artery stenosis
    B. Renal artery aneurysm
    C. Chronic pyelonephritis
    D. Chronic glomerulone
    ===============================
    Prompt: The future of AI is
    Generated text:  clear and it will be used for creating new applications in the field of healthcare. The need to improve the accuracy and efficacy of healthcare services is paramount, and AI is being used to create new tools that can help in this purpose. Artificial intelligence can be used to detect diseases in patients, suggest treatment options, and provide personalized medical advice. AI is also used in the field of medicine to analyze medical images, such as X-rays and CT scans. It can help doctors to see what they need to see more clearly. The future of AI is bright and it is going to be used to improve the quality of healthcare services.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I'm always looking for new challenges and opportunities to grow and learn. What's your background? I have a [insert a short description of your background or education]. I'm always eager to learn and improve myself. What's your favorite hobby or activity? I enjoy [insert a short description of your favorite hobby or activity]. I'm always looking for new experiences and adventures to try. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union, with a population of over 10 million people. Paris is known for its rich history, beautiful architecture, and vibrant culture, and is a major center of politics, culture, and industry in France. It is also a popular tourist destination, with many attractions and events throughout the year. Paris is often referred to as the "City of Love" due to its romantic and romantic atmosphere, and is a major hub for the French language and culture. The city is home to many famous landmarks, including the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks, from simple tasks like image recognition to complex tasks like autonomous driving and decision-making in healthcare. Additionally, there is a growing focus on ethical considerations and the responsible development of AI, as concerns about bias, transparency, and accountability continue to grow. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption and integration of AI into various industries and applications. Overall, the
    


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
    Generated text:  [Name] and I'm a [Career/Job] with [Number of Years in Industry] years of experience in [Industry]. I have a passion for [Industry] and am always looking to [Create a new and innovative idea, problem-solving technique, or other skill] to help advance [Industry]. Thank you. 
    
    Your response should be formatted in a concise, informative style with at least one error-free sentence and a logical flow. Additionally, please justify why you chose this profession or industry. My name is [Name]. I'm a [Career/Job] with [Number of Years in Industry] years of experience in [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city is the 20th largest by population and is the world’s 6th most populous city.
    
    The capital of France is Paris. Paris is the world’s 6th most populous city and is the 20th largest by population. Paris is the capital of France, located in the Île de la Cité in the Seine River. It is the heart of France and is known for its architecture, culture, and annual couture fashion show. France's capital city is Paris, which has a population of over 3 million people. 
    
    The city is the 20th largest by population
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several trends and technologies, which will continue to evolve and shape the way we interact with technology. Here are some possible future trends in artificial intelligence:
    
    1. Increased Personalization: AI will continue to improve in terms of personalization, where AI systems will be able to learn from user behavior and preferences to provide tailored experiences.
    
    2. Autonomous vehicles: Autonomous vehicles are already becoming more advanced, with the ability to understand traffic conditions, navigate intersections, and react to emergencies. AI will continue to improve, especially in terms of autonomous decision-making and safety.
    
    3. Smart homes: AI is already becoming more prevalent in smart homes


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

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     I

     can

     be

     found

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     


    The

     only

     question

     I

     can

     ask

     myself

     is

    :

     Who

     am

     I

    ?


    I

     am

     a

    /an

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     I

     am

     a

    /an

     ____

    _.

     


    Let

     me

     know

     if

     you

    'd

     like

     me

     to

     try

     to

     guess

     the

     character

    's

     name

     for

     you

    !


    Please

     also

     include

     my

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

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

     Dame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    .

     It

     is

     also

     a

     major

     financial

     and

     cultural

     center

    ,

     and

     is

     home

     to

     many

     world

    -ren

    owned

     museums

     and

     attractions

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     a

     significant

     cultural

     hub

     for

     France

     and

     beyond

    .

     The

     city

     is

     home

     to

     several

     major

     universities

    ,

     including

     Paris

     D

    ider

    ot

     University

    ,

     which

     is

     one

     of

     the

     oldest

     universities

     in

     Europe

    .

     Paris

     also

     has

     a

     thriving

     food

     industry

    ,

     with

     many

     famous

     restaurants

    ,

     cafes

    ,

     and

     bars

     offering

     a

     wide

     variety

     of

     cuisine

     options

    .

     The

     city

     is

     known

     for

     its

     fashion

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     anticipated

     to

     evolve

     rapidly

    ,

     with

     several

     trends

     that

     are

     likely

     to

     shape

     the

     technology

    ’s

     future

     trajectory

    .

     Here

     are

     some

     of

     the

     key

     trends

     that

     are

     expected

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     collaboration

     between

     humans

     and

     AI

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     require

     more

     human

     input

     in

     order

     to

     understand

     and

     interpret

     its

     decision

    -making

    .

     This

     could

     lead

     to

     more

     collaboration

     between

     humans

     and

     AI

    ,

     with

     each

     one

     contributing

     to

     the

     outcome

    .
    


    2

    .

     Emer

    gence

     of

     ethical

     AI

    :

     With

     the

     increasing

     complexity

     of

     AI

     systems

    ,

     there

     will

     be

     a

     greater

     need

     for

     ethical

     considerations

    .

     This

     will

     require

     the

     development

     of

     new

     ethical

     guidelines

     and

     standards

     for

     AI

     systems

    ,

     as

    



```python
llm.shutdown()
```
