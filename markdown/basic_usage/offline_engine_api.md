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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.93it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.93it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.93it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.79it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.79it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.79it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.79it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.79it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.79it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.79it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.79it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.79it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.79it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.79it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.61it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.61it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.61it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.61it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.61it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.61it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.61it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.61it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.61it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.61it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.61it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.40it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.61it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.61it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   7%|▋         | 4/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   7%|▋         | 4/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.54it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.36it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.36it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.54it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.72it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.72it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.72it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.72it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.72it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.72it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.19it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.19it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.19it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.19it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.19it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.19it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.34it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  71%|███████   | 41/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  71%|███████   | 41/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.24it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.24it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.28it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.28it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.28it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.28it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.28it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.27it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.27it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.27it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 37.27it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 36.40it/s]


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
    Generated text:  Jessica and I live in Austin, Texas. I am a senior at University of Texas at Austin. I have a degree in Accounting and Finance. I graduated in 2017 with a strong performance in the Western History course. I am currently majoring in Business Administration. I enjoy being outdoors, hiking, skiing, and watching sports. I also really like to travel to new places and explore new cultures. Can you recommend some good restaurants in Austin for a business trip? Jessica is doing great work, but I need a bit more information to make a recommendation. Can you provide more details on the types of restaurants she enjoys and the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking government official who is typically appointed by the head of state and confirmed by the United States Senate. Some may argue that the person who is in charge of the executive branch of the federal government is not, in fact, the president, since the president is elected to the office by the national legislature. Does this mean that the president is not the head of state of the United States? No, the president is not the head of state of the United States. While the president is a high-ranking government official who is typically appointed by the head of state and confirmed by the United States Senate, the president does not have the same standing
    ===============================
    Prompt: The capital of France is
    Generated text:  _____. The capital of France is Paris. Paris is known as the "city of love" because of its famous beauty.
    
    The answer is: Paris. 
    
    To elaborate, Paris is the capital of France, and it is renowned for its beautiful city skyline and iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city's romantic atmosphere and charm are a result of its historical significance and cultural richness, which are reflected in its name. The city is also known for its annual Eiffel Tower walk, which celebrates its 100th anniversary. 
    
    The capital of France is indeed
    ===============================
    Prompt: The future of AI is
    Generated text:  under attack, and it is widely known that AI is one of the most promising fields in science and technology. Although the field has seen significant progress in recent years, there is still a lot of room for improvement. Therefore, the future of AI is very promising, but it will be a long road ahead of us to achieve our goals. In order to develop AI, we need to overcome some challenges and make necessary improvements to prevent any potential risks. When it comes to AI, we must believe that it can be a positive and valuable tool for us to use in our daily lives and work. Therefore, we need to be mindful of the


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am passionate about [reason for being at the company]. I am always looking for ways to [what I enjoy doing at the company]. I am [age] years old and I have [number of years of experience] years of experience in [industry]. I am [gender] and I am [height] inches tall. I have [weight] pounds. I am [eye color] and I have [hair color]. I am [gender] and I have [gender] hair. I am [gender] and I have [gender] eyes.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter, a historic neighborhood. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also home to the French Riviera, a coastal area with beautiful beaches and Mediterranean cuisine. The city is known for its fashion industry, with iconic fashion houses such as Chanel and Louis Vuitton. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine. Additionally, AI is likely to play an increasingly important role in shaping the future of work, with the rise of automation and artificial intelligence becoming more prevalent in industries such as manufacturing, finance, and healthcare. AI is also likely to have a significant impact on society, with the potential to address some of the most pressing challenges facing humanity, such as climate change and global health. Finally, AI is likely to continue
    


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
    Generated text:  [Name] and I'm a [Position] for [Company]. I'm excited to be here today and I'm looking forward to [My First Day at [Company]]. Thank you for taking the time to meet me.
    
    It seems like you are the company's CEO or CEO of the company. That's great! Can you tell me a bit more about your background and what you bring to the table at the company? I'm curious to know more about your journey and what you have accomplished.
    
    I'm excited to learn more about you and your background at the company. Thank you for sharing your story. Is there anything in particular
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. 
    
    While the city is home to numerous museums and art galleries, the main attraction is the Notre-Dame Cathedral, which is considered a symbol of French culture and history. The city also boasts a rich culinary scene with famous dishes like croissants and macarons, and is home to the Eiffel Tower, a famous landmark. 
    
    Paris is known for its diverse neighborhoods, including the Latin Quarter, which is steeped in history and culture, and the Seine River, which is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see a wide range of technological advancements, both positive and negative. Here are some possible future trends in AI:
    
    1. AI will continue to develop and improve, leading to more efficient, accurate, and personalized applications.
    
    2. AI will be integrated into more and more everyday technologies, such as healthcare, transportation, and consumer products.
    
    3. AI will be used to improve human skills, such as language translation, emotional intelligence, and data analysis.
    
    4. AI will be used to solve complex problems and make decisions based on vast amounts of data.
    
    5. AI will continue to be used for both personal and business applications, with more


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

    ].

     I

     am

     a

     writer

    ,

     illustrator

    ,

     and

     painter

    .

     My

     hobbies

     include

     photography

    ,

     yoga

    ,

     and

     playing

     board

     games

    .

     I

     enjoy

     experimenting

     with

     new

     mediums

    ,

     learning

     new

     techniques

    ,

     and

     trying

     different

     approaches

     to

     storytelling

    .

     I

     love

     to

     share

     my

     work

     with

     others

     and

     am

     always

     looking

     for

     new

     ways

     to

     improve

     my

     craft

    .

     I

     am

     a

     fan

     of

     [

    genre

    ]

     and

     love

     to

     explore

     new

     writing

     styles

     and

     techniques

    .

     I

     am

     a

     confident

    ,

     artistic

    ,

     and

     creative

     person

     who

     is

     always

     looking

     for

     ways

     to

     express

     myself

     through

     my

     art

    .

     I

     am

     a

     natural

     at

     catching

     the

     eye

     of

     readers

     and

     am

     always

     eager

     to

     create

     new

     and

     engaging

     stories

     for

     my

     readers

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     often

     referred

     to

     as

     the

     "

    City

     of

     a

     Thousand

    言

    "

     due

     to

     its

     sheer

     volume

     of

     speakers

    .

     
    


    (Note

    :

     "

    言

    "

     in

     the

     context

     of

     language

     refers

     to

     the

     number

     of

     people

     speaking

     a

     particular

     language

    .)
    


    Paris

     is

     situated

     on

     the

     left

     bank

     of

     the

     Se

    ine

     River

    ,

     near

     the

     city

     of

     Paris

     itself

    ,

     while

     the

     Se

    ine

     River

     flows

     through

     the

     center

     of

     the

     city

    .

     The

     city

     is

     known

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

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Notre

     Dame

     Basil

    ica

    .

     Paris

     is

     also

     a

     popular

     tourist

     destination

    ,

     hosting

     numerous

     cultural

     and

     artistic

     events

     throughout

     the

     year

    .

     
    


    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     exciting

     developments

    .

     Some

     potential

     trends

     include

    :
    


    1

    .

     Integration

     with

     other

     technologies

    :

     AI

     will

     likely

     become

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

    ,

     bi

    otechnology

    ,

     and

     data

     analytics

    ,

     to

     improve

     their

     effectiveness

     and

     accuracy

    .
    


    2

    .

     Greater

     autonomy

     and

     flexibility

    :

     AI

     will

     become

     more

     autonomous

     and

     flexible

    ,

     allowing

     it

     to

     learn

     and

     adapt

     to

     new

     situations

     and

     improve

     its

     performance

     over

     time

    .
    


    3

    .

     Integration

     with

     human

     workers

    :

     AI

     will

     be

     integrated

     with

     human

     workers

     to

     increase

     productivity

     and

     efficiency

    ,

     reducing

     the

     need

     for

     manual

     labor

    .
    


    4

    .

     Increased

     integration

     with

     human

     emotions

    :

     AI

     will

     become

     more

     integrated

     with

     human

     emotions

    ,

     allowing

     it

     to

     understand

     and

     empath

    



```python
llm.shutdown()
```
