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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:17,  2.87it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.77it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.77it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.77it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.77it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.77it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.77it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.77it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.77it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.77it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.31it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.31it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.31it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.31it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.31it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.31it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.31it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 12.31it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 12.31it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:04<00:02, 12.31it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 19.77it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 19.77it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 19.77it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 19.77it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:01, 19.77it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:04<00:01, 19.77it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:04<00:01, 19.77it/s]

    Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:04<00:01, 19.77it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:04<00:01, 19.77it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:04<00:01, 19.77it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 28.10it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 28.10it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 28.10it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 28.10it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 28.10it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 28.10it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 28.10it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 28.10it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 28.10it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 28.10it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 36.77it/s]

    Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 36.77it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 36.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.69 GB):   3%|▎         | 2/58 [00:00<00:03, 14.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.68 GB):   3%|▎         | 2/58 [00:00<00:03, 14.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=69.68 GB):   3%|▎         | 2/58 [00:00<00:03, 14.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.68 GB):   7%|▋         | 4/58 [00:00<00:03, 15.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.68 GB):   7%|▋         | 4/58 [00:00<00:03, 15.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.67 GB):   7%|▋         | 4/58 [00:00<00:03, 15.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.67 GB):  10%|█         | 6/58 [00:00<00:03, 14.86it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.66 GB):  10%|█         | 6/58 [00:00<00:03, 14.86it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=69.66 GB):  10%|█         | 6/58 [00:00<00:03, 14.86it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.66 GB):  10%|█         | 6/58 [00:00<00:03, 14.86it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.66 GB):  10%|█         | 6/58 [00:00<00:03, 14.86it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.66 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.65 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.15it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=69.65 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.65 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.65 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.02it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.63 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.63 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.02it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.63 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.02it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=69.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.49it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=69.61 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.49it/s]Capturing num tokens (num_tokens=960 avail_mem=69.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.49it/s] Capturing num tokens (num_tokens=896 avail_mem=69.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.49it/s]Capturing num tokens (num_tokens=832 avail_mem=69.61 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.49it/s]Capturing num tokens (num_tokens=832 avail_mem=69.61 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=768 avail_mem=69.61 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=704 avail_mem=69.61 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=640 avail_mem=69.60 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=576 avail_mem=69.60 GB):  41%|████▏     | 24/58 [00:01<00:01, 33.50it/s]Capturing num tokens (num_tokens=512 avail_mem=69.59 GB):  41%|████▏     | 24/58 [00:01<00:01, 33.50it/s]Capturing num tokens (num_tokens=512 avail_mem=69.59 GB):  50%|█████     | 29/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=480 avail_mem=69.60 GB):  50%|█████     | 29/58 [00:01<00:00, 37.18it/s]

    Capturing num tokens (num_tokens=448 avail_mem=69.60 GB):  50%|█████     | 29/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=416 avail_mem=69.60 GB):  50%|█████     | 29/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=384 avail_mem=69.60 GB):  50%|█████     | 29/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=352 avail_mem=69.59 GB):  50%|█████     | 29/58 [00:01<00:00, 37.18it/s]Capturing num tokens (num_tokens=352 avail_mem=69.59 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=320 avail_mem=69.59 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=288 avail_mem=69.58 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=256 avail_mem=69.58 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=240 avail_mem=69.58 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=224 avail_mem=69.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=224 avail_mem=69.57 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.80it/s]Capturing num tokens (num_tokens=208 avail_mem=69.57 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.80it/s]

    Capturing num tokens (num_tokens=192 avail_mem=69.57 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.80it/s]Capturing num tokens (num_tokens=176 avail_mem=69.57 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.80it/s]Capturing num tokens (num_tokens=160 avail_mem=69.57 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.80it/s]Capturing num tokens (num_tokens=144 avail_mem=69.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.80it/s]Capturing num tokens (num_tokens=144 avail_mem=69.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=128 avail_mem=69.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=112 avail_mem=69.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=96 avail_mem=69.55 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.69it/s] Capturing num tokens (num_tokens=80 avail_mem=69.55 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=64 avail_mem=69.55 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=64 avail_mem=69.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.60it/s]

    Capturing num tokens (num_tokens=48 avail_mem=69.54 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=32 avail_mem=69.54 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=28 avail_mem=69.53 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=24 avail_mem=69.53 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=20 avail_mem=69.53 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=20 avail_mem=69.53 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=16 avail_mem=69.53 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=12 avail_mem=69.52 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=8 avail_mem=69.52 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.17it/s] Capturing num tokens (num_tokens=4 avail_mem=69.51 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=4 avail_mem=69.51 GB): 100%|██████████| 58/58 [00:01<00:00, 34.25it/s]


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
    Generated text:  Fahad Khan. I’m a Computer Science and Engineering major at Georgia Tech. I majored in Software Engineering, and my studies have been instrumental in preparing me for a career in the tech industry. I have a passion for programming, and I have a strong drive to learn and develop new skills.
    I am a member of the hackathon team for Georgia Tech and have been involved in numerous competitions and hackathons. I have been selected for multiple teams to compete against other students and have been recognized for my exceptional performance. I have also been invited to participate in hackathons and coding competitions for several years, and I have been recognized
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by a vice president. The vice president is represented by a secretary. The secretary is represented by the first lady. If each position has 50% more attendees than the previous one, how many people are there in total?
    To determine the total number of people representing the offices, we need to understand the pattern of the numbers and calculate accordingly.
    
    1. The president of the United States is represented by a vice president.
    2. The vice president is represented by a secretary.
    3. The secretary is represented by the first lady.
    
    We observe that the roles are hierarchical:
    - President (H)
    - Vice President (V)
    -
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is also known as the "city of love". The city has a population of about 1,580,000 inhabitants. What is the scientific notation of this number? To convert the number 1,580,000 into scientific notation, follow these steps:
    
    1. Identify the significant digits in the number. In this case, they are 1.58.
    2. Determine the power of 10 that will move the decimal point. To do this, count the number of places the decimal point needs to move from its original position in 1,580,
    ===============================
    Prompt: The future of AI is
    Generated text:  very promising, but it also presents a number of challenges. Will it bring benefits to society in the long run, or will it result in the loss of jobs? In this article, we will explore the potential benefits and challenges of AI, and consider what can be done to mitigate them. We will also examine potential solutions to the potential negatives of AI, including the need for a shift towards more sustainable and ethical practices in the use of AI. We hope that this article can provide some insight into the complex and dynamic landscape of AI, and help to guide the future of technology as a whole. #AI #future #challenges #benef


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I've been with the company for [number of years] years. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] at [company name], and I'm excited to help you achieve your goals. What's your name, and what's your job title? [Name] [Job Title] [Company Name] [Company Address] [Company Phone Number] [Company Email] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and diverse culture. It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 8 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe, as well as its cuisine, fashion, and music. It is a major center of art, culture, and science, and is a UNESCO World Heritage site. Paris is a popular tourist destination and a major economic and financial center
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be an increased focus on ethical considerations. This includes issues such as bias, transparency, accountability, and privacy. As a result, there will be a push for more robust ethical guidelines and standards for AI development and deployment.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to new situations. This could lead to more efficient and
    


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
    Generated text:  [Name] and I am a friendly and helpful individual who always tries to provide positive support to those around me. I'm passionate about helping people and I have a keen interest in sustainability and environmental conservation. I'm confident and competent, and I thrive on learning and staying up-to-date with the latest trends and technologies. I'm a true resourceful and resourceful individual who is always ready to assist with whatever task or problem I come across. I'm friendly, energetic, and I enjoy interacting with people of all ages and backgrounds. I believe that everyone can make a positive impact and that I'm up for the challenge of helping others in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a UNESCO World Heritage site and one of the most cosmopolitan cities in the world. Paris is known for its rich history, stunning architecture, and diverse cultural scene. It is the home of the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral, among other iconic landmarks. The city is also a major transportation hub, with many famous landmarks, including the Champs-Elysées and the Seine River. Paris is a vibrant and dynamic city, and it has a rich cultural heritage that continues to inspire people from all over the world. Its status as the capital has made Paris one of the most
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and dynamic, with many different areas and technologies that are likely to shape the field in the coming years. Here are some of the possible trends in AI:
    
    1. Improved Accuracy: One of the most promising areas of AI development is the improvement of accuracy and precision in computer vision and other AI applications. With advancements in neural networks and other machine learning techniques, it is possible that AI systems will become more accurate and precise in their predictions and decisions.
    
    2. Integration with Human Behavior: AI systems are becoming more and more integrated with human behavior. As AI systems become more capable of understanding and predicting human behavior, they could be used to improve


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

     am

     a

     [

    Job

     Title

    ].

     I

     have

     always

     been

     passionate

     about

     [

    field

     of

     interest

    ]

     and

     I

     have

     always

     been

     looking

     forward

     to

     [

    one

     or

     more

     activities

     or

     projects

     related

     to

     this

     field

     of

     interest

    ].

     I

     have

     been

     making

     some

     progress

     in

     [

    field

     of

     interest

    ]

     and

     I

     am

     eager

     to

     continue

     learning

     more

     and

     growing

     my

     skills

    .

     If

     you

     have

     any

     questions

     or

     need

     any

     help

    ,

     please

     don

    't

     hesitate

     to

     contact

     me

    .

     Thank

     you

    .

     [

    Name

    ]

     [

    Contact

     Information

    ]

     [

    Date

    ]

     [

    Address

    ]

     [

    Phone

     Number

    ]

     [

    Email

     Address

    ]

     [

    Website

    ]

     [

    Social

     Media

     Handles

    ]

     [

    Job

     Description

    ]


    [

    Job

     Title

    ]


    My

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    In

     addition

     to

     its

     historical

     significance

    ,

     Paris

     is

     a

     vibrant

     city

     known

     for

     its

     iconic

     landmarks

    ,

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

    .

     The

     city

     is

     also

     renowned

     for

     its

     rich

     cultural

     heritage

    ,

     with

     many

     museums

    ,

     theaters

    ,

     and

     galleries

     showcasing

     world

    -class

     art

     and

     architecture

    .

     
    


    Paris

     is

     also

     home

     to

     a

     diverse

     population

    ,

     with

     a

     mix

     of

     immigrant

     groups

     and

     residents

    ,

     including

     French

    ,

     Spanish

    ,

     English

    -speaking

    ,

     and

     other

     cultures

    .

     It

     has

     a

     rich

     cultural

     scene

    ,

     with

     many

     events

    ,

     festivals

    ,

     and

     concerts

     taking

     place

     throughout

     the

     year

    .

     
    


    Overall

    ,

     Paris

     is

     a

     bustling

     and

     fascinating

     city

     that

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     exciting

     and

     unpredictable

    ,

     and

     it

     is

     possible

     to

     see

     many

     different

     trends

     developing

     in

     the

     coming

     years

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Improved

     accuracy

     and

     reliability

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

     even

     greater

     accuracy

     and

     reliability

     in

     the

     algorithms

     used

     to

     process

     and

     analyze

     data

    .

     This

     could

     lead

     to

     more

     accurate

     predictions

    ,

     more

     effective

     decision

    -making

    ,

     and

     ultimately

    ,

     more

     productive

     outcomes

    .
    


    2

    .

     Increased

     human

     involvement

    :

     While

     AI

     is

     becoming

     more

     capable

    ,

     there

     is

     a

     possibility

     that

     it

     will

     continue

     to

     evolve

     alongside

     humans

     and

     play

     a

     more

     significant

     role

     in

     decision

    -making

     and

     decision

    -making

    .

     This

     could

     result

     in

     more

     complex

     and

    



```python
llm.shutdown()
```
