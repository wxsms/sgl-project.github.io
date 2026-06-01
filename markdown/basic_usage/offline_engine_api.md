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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:44,  4.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:44,  4.99s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:44,  4.99s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:44,  4.99s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:44,  4.99s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.29it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.29it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.29it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.12it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 18.93it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 18.93it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 18.93it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 18.93it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 18.93it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 18.93it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 18.93it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 18.93it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 18.93it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 18.93it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 26.75it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 26.75it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 26.75it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 26.75it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 26.75it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 26.75it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 26.75it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 26.75it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 26.75it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 26.75it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 35.31it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 35.31it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 35.31it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 35.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 28.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 28.27it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 28.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 28.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 28.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.75it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.14it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.14it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.14it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.14it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.14it/s]

    Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.73it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.73it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.73it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.73it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.73it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.73it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.92it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.92it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.92it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.92it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 36.92it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 36.92it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  60%|██████    | 35/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.02it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.73it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.24it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.26it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.26it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.26it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 36.26it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 35.02it/s]


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
    Generated text:  Asha and I'm a 16 year old female who is currently in college. I recently started my freshman year and am starting to develop ulcers on my lower lip. The ulcers have been there for about 2 months and they've gotten bigger and more scaly. I've been experiencing pain and itching on my lower lip as well. I'm pretty sure they're ulcers and I'm not sure if this is what's causing the pain and itching. I've done a bit of research online on the internet, but I'm not entirely sure. 
    
    Should I consult a doctor to get an accurate diagnosis and treatment plan
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office. The president is elected every four years. In the last four years, the first president was elected in 2009, the second in 2013, the third in 2017, and the fourth in 2021. How many more years are left before the fifth president is elected?
    To determine how many more years are left before the fifth president is elected, we need to calculate the total number of years that have passed since the first president was elected in 2009, and then subtract this from the total number of years that remain until the next election.
    
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the most important city in France, and a major city in Europe. Located in the north central part of the country, it is situated on the banks of the Seine river. With a population of over 2 million, it is the second most populous city in France, and the sixth most populous city in the world, after Tokyo. The city has a long and rich history of being a major hub for commerce and culture in Europe.
    
    The city has a population of over 2 million, and it is the most populous city in France, and the sixth most populous city in the world after Tokyo. It is the
    ===============================
    Prompt: The future of AI is
    Generated text:  _____. It will provide more detailed and accurate images, that will allow humans to better understand the world around them.
    A. visionary
    B. revolutionary
    C. extravagant
    D. amazing
    Answer:
    B
    
    When using the "64x64" display size, what is the display ratio?
    A. 1:1
    B. 1:2
    C. 2:1
    D. 3:1
    Answer:
    A
    
    What is the most critical task of an operating system?
    A. Processing data
    B. Managing and controlling the entire computer system
    C. Handling input and output
    D


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill] who has [Number of Years] years of experience in [Field of Interest]. I'm a [Skill] who has [Number of Years] years of experience in [Field of Interest]. I'm a [Skill] who has [Number of Years] years of experience in [Field of Interest]. I'm a [Skill] who has [Number of Years] years of experience in [Field of Interest]. I'm a [Skill] who has [Number of Years] years of experience in [Field of Interest
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its fashion industry, art scene, and its role in French politics and diplomacy. It is also home to many famous French chefs and is known for its cuisine. Paris is a city that has a unique blend of history, culture, and modernity, making it a popular destination for tourists and locals alike. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential future trends in AI:
    
    1. Increased automation and robotics: As AI continues to improve, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency, reduced costs, and improved quality of life.
    
    2. AI-powered healthcare: AI will play a key role in improving the quality of healthcare by providing more accurate diagnoses, personalized treatment plans, and improved patient outcomes. AI-powered healthcare systems will also be able to analyze
    


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
    Generated text:  [Your Name] and I am [Your Age]. I am a [Your Profession/Job Title] with a passion for [Your Field/Interest]. I am a [Your Country] native and have been [Your Career Path/Experience] since [Your Date of Birth]. I have always been interested in [Your Hobby/Interest], and I am always looking for new challenges to push myself. I am [Your Personality/Tone/Adjective]! I am always looking to learn new things and have a good sense of humor. I like to make people laugh and make them smile, so I believe that I can make a difference
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city where the Eiffel Tower, the Louvre Museum, and the Palace of Versailles once stood. It is the most populous city in France and the second-largest city in Europe. Paris is renowned for its rich culture, history, and cuisine, including its annual Bastille Day celebrations and annual fashion week. The city is known for its iconic landmarks, including the Eiffel Tower, the Louvre Museum, and the Arc de Triomphe. Paris is also the birthplace of countless notable figures in French culture and history, including Victor Hugo and Jean-Paul Sartre. Finally, Paris is known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dynamic and is likely to evolve rapidly as new technologies and applications emerge. Here are some possible trends in AI that could shape the future of the field:
    
    1. Increased automation: AI is already becoming more automated, with self-driving cars, robots, and automation in manufacturing and healthcare. AI is expected to become even more integrated with the labor market, leading to a shift towards a more automated and less human-centric future.
    
    2. Improved interpretability: AI systems are becoming more transparent and interpretable, with more detailed explanations of their decisions and actions. This could lead to increased trust and confidence in AI systems, but it could also raise ethical


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

     name

    ]

     and

     I

     am

     a

     [

    insert

     profession

    ],

     [

    insert

     field

    ].

     My

     work

     is

     to

     ensure

     the

     sustainability

     of

     [

    insert

     topic

    ].

     I

     am

     passionate

     about

     [

    insert

     topic

    ],

     as

     it

     is

     essential

     to

     ensure

     that

     our

     planet

     and

     its

     resources

     can

     continue

     to

     support

     human

     life

    .

     I

     am

     also

     dedicated

     to

     promoting

     environmental

     consciousness

     and

     sustainable

     living

    .

     If

     you

     have

     any

     questions

     or

     need

     assistance

    ,

     don

    't

     hesitate

     to

     reach

     out

    .

     I

     look

     forward

     to

     the

     possibility

     of

     working

     with

     you

    .

     [

    insert

     name

    ]

     I

     am

     passionate

     about

     sustainability

     and

     environmental

     consciousness

    .

     [

    insert

     name

    ]

     I

     am

     a

     [

    insert

     profession

    ],

     [

    insert

     field

    ].

     I

     specialize

     in

     [

    insert

     topic

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Explanation

     for

     a

     

    6

    -year

    -old

    :

     Paris

     is

     the

     big

     city

     where

     all

     the

     big

     important

     places

     are

    .

     It

    's

     like

     the

     super

     important

     building

     in

     the

     whole

     world

    .

     
    


    F

    acts

     to

     remember

    :


    -

     Paris

     is

     the

     capital

     of

     France

    


    -

     It

     has

     lots

     of

     big

     buildings

     and

     yummy

     food

    


    -

     It

    's

     always

     full

     of

     people

     and

     fun

     places

     to

     visit

    !

     
    


    The

     word

     "

    capital

    "

     means

     it

    's

     the

     main

     or

     most

     important

     city

    .

     This

     city

     helps

     run

     and

     help

     people

     in

     France

    .

     It

    's

     like

     a

     big

     house

     for

     the

     whole

     country

    !

     


    (R

    em

    ember

     to

     use

     friendly

     words

     like

     "

    big

     city

    "

     and

     "

    important

     place

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     key

     trends

    :
    


    1

    .

     Increased

     efficiency

     and

     accuracy

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     it

     is

     expected

     to

     become

     more

     efficient

     and

     accurate

     in

     solving

     complex

     problems

    .

     This

     could

     lead

     to

     faster

     decision

    -making

     and

     improved

     outcomes

     for

     various

     applications

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .
    


    2

    .

     Integration

     with

     human

     decision

    -making

    :

     AI

     is

     likely

     to

     become

     more

     integrated

     with

     human

     decision

    -making

     processes

    ,

     with

     the

     ability

     to

     consider

     multiple

     factors

     and

     perspectives

     in

     making

     decisions

    .

     This

     could

     lead

     to

     more

     informed

     and

     balanced

     decisions

     that

     take

     into

     account

     a

     wider

     range

     of

     variables

    .
    


    3

    .

     Privacy

     and

     security

     concerns

    :

     As

     AI

     becomes

     more

     prevalent

     in

     our

     lives

    



```python
llm.shutdown()
```
