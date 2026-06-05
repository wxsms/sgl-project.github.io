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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.35it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.20it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.20it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.41it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.41it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.41it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.41it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.41it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.41it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.41it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.41it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.74it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.90 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.87 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.87 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.86 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.86 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.86 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.86 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.86 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.85 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.84 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.84 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.84 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.84 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.83 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.83 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.83 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=55.83 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.18it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.83 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.82 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.82 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.81 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.81 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.81 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.81 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.81 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.79 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s]Capturing num tokens (num_tokens=960 avail_mem=55.80 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s] Capturing num tokens (num_tokens=896 avail_mem=55.80 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s]

    Capturing num tokens (num_tokens=832 avail_mem=55.80 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s]Capturing num tokens (num_tokens=832 avail_mem=55.80 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.30it/s]Capturing num tokens (num_tokens=768 avail_mem=55.79 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.30it/s]Capturing num tokens (num_tokens=704 avail_mem=55.79 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.30it/s]Capturing num tokens (num_tokens=640 avail_mem=55.79 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.30it/s]Capturing num tokens (num_tokens=576 avail_mem=55.79 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.30it/s]Capturing num tokens (num_tokens=512 avail_mem=55.77 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.30it/s]Capturing num tokens (num_tokens=512 avail_mem=55.77 GB):  50%|█████     | 29/58 [00:00<00:00, 43.66it/s]Capturing num tokens (num_tokens=480 avail_mem=55.79 GB):  50%|█████     | 29/58 [00:00<00:00, 43.66it/s]Capturing num tokens (num_tokens=448 avail_mem=55.78 GB):  50%|█████     | 29/58 [00:00<00:00, 43.66it/s]Capturing num tokens (num_tokens=416 avail_mem=55.78 GB):  50%|█████     | 29/58 [00:00<00:00, 43.66it/s]Capturing num tokens (num_tokens=384 avail_mem=55.78 GB):  50%|█████     | 29/58 [00:00<00:00, 43.66it/s]

    Capturing num tokens (num_tokens=352 avail_mem=55.77 GB):  50%|█████     | 29/58 [00:00<00:00, 43.66it/s]Capturing num tokens (num_tokens=352 avail_mem=55.77 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.34it/s]Capturing num tokens (num_tokens=320 avail_mem=55.77 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.34it/s]Capturing num tokens (num_tokens=288 avail_mem=55.77 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.34it/s]Capturing num tokens (num_tokens=256 avail_mem=55.76 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.34it/s]

    Capturing num tokens (num_tokens=240 avail_mem=55.76 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=224 avail_mem=55.76 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.34it/s]Capturing num tokens (num_tokens=224 avail_mem=55.76 GB):  67%|██████▋   | 39/58 [00:01<00:00, 31.36it/s]Capturing num tokens (num_tokens=208 avail_mem=55.75 GB):  67%|██████▋   | 39/58 [00:01<00:00, 31.36it/s]Capturing num tokens (num_tokens=192 avail_mem=55.75 GB):  67%|██████▋   | 39/58 [00:01<00:00, 31.36it/s]Capturing num tokens (num_tokens=176 avail_mem=55.75 GB):  67%|██████▋   | 39/58 [00:01<00:00, 31.36it/s]Capturing num tokens (num_tokens=160 avail_mem=55.75 GB):  67%|██████▋   | 39/58 [00:01<00:00, 31.36it/s]

    Capturing num tokens (num_tokens=160 avail_mem=55.75 GB):  74%|███████▍  | 43/58 [00:01<00:00, 31.47it/s]Capturing num tokens (num_tokens=144 avail_mem=55.72 GB):  74%|███████▍  | 43/58 [00:01<00:00, 31.47it/s]Capturing num tokens (num_tokens=128 avail_mem=55.72 GB):  74%|███████▍  | 43/58 [00:01<00:00, 31.47it/s]Capturing num tokens (num_tokens=112 avail_mem=55.70 GB):  74%|███████▍  | 43/58 [00:01<00:00, 31.47it/s]Capturing num tokens (num_tokens=96 avail_mem=55.69 GB):  74%|███████▍  | 43/58 [00:01<00:00, 31.47it/s] Capturing num tokens (num_tokens=96 avail_mem=55.69 GB):  81%|████████  | 47/58 [00:01<00:00, 32.05it/s]Capturing num tokens (num_tokens=80 avail_mem=55.69 GB):  81%|████████  | 47/58 [00:01<00:00, 32.05it/s]Capturing num tokens (num_tokens=64 avail_mem=55.68 GB):  81%|████████  | 47/58 [00:01<00:00, 32.05it/s]

    Capturing num tokens (num_tokens=48 avail_mem=55.68 GB):  81%|████████  | 47/58 [00:01<00:00, 32.05it/s]Capturing num tokens (num_tokens=32 avail_mem=55.67 GB):  81%|████████  | 47/58 [00:01<00:00, 32.05it/s]Capturing num tokens (num_tokens=32 avail_mem=55.67 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.87it/s]Capturing num tokens (num_tokens=28 avail_mem=55.67 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.87it/s]Capturing num tokens (num_tokens=24 avail_mem=55.67 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.87it/s]Capturing num tokens (num_tokens=20 avail_mem=55.66 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.87it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.66 GB):  93%|█████████▎| 54/58 [00:01<00:00, 25.30it/s]Capturing num tokens (num_tokens=16 avail_mem=55.66 GB):  93%|█████████▎| 54/58 [00:01<00:00, 25.30it/s]Capturing num tokens (num_tokens=12 avail_mem=55.66 GB):  93%|█████████▎| 54/58 [00:01<00:00, 25.30it/s]Capturing num tokens (num_tokens=8 avail_mem=55.65 GB):  93%|█████████▎| 54/58 [00:01<00:00, 25.30it/s] Capturing num tokens (num_tokens=8 avail_mem=55.65 GB):  98%|█████████▊| 57/58 [00:01<00:00, 23.80it/s]Capturing num tokens (num_tokens=4 avail_mem=55.65 GB):  98%|█████████▊| 57/58 [00:01<00:00, 23.80it/s]Capturing num tokens (num_tokens=4 avail_mem=55.65 GB): 100%|██████████| 58/58 [00:01<00:00, 30.03it/s]


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
    Generated text:  David. I'm a software engineer who has a passion for discovering new programming languages. I currently work for the company "Laurent Guerzhoy". Can you tell me about your programming languages and programming languages you love to learn about? Are you also passionate about coding challenges and contests? Yes, as a software engineer, I love to discover new programming languages and learning new things. I also love coding challenges and contests as they help me stay up-to-date with the latest programming trends and technologies. Additionally, I enjoy sharing my knowledge with others and helping them learn new programming languages. How about you? Do you have any favorite programming languages
    ===============================
    Prompt: The president of the United States is
    Generated text:  interested in learning the number of students in his administration. He asks his assistant to count the students. The assistant starts with a total of 300 students. He then asks if there are any students in the office. If the assistant says "no," he says "good," and if he says "yes," he says "bad." The president then tells him that the assistant made the same mistake twice. If the assistant said "bad" twice, how many students did the assistant actually count in the office? To determine the number of students the assistant actually counted in the office, we need to analyze the situation step by step.
    
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It was founded on 6 January 1137 and is located on the banks of the Seine river. It is a huge city with a population of over 2 million people, and is often referred to as the 'City of Love'.
    
    The city has been a French capital since 900 AD. The city was founded by a small group of French nobles and French monks to be the seat of the French kingdom. In 1137, the city was named after a French king called Louis I, who was the founder of the dynasty that ruled France.
    
    The city has been home to many
    ===============================
    Prompt: The future of AI is
    Generated text:  bright: Machine learning is getting better at predicting future events and helping to build safer, more efficient, more efficient, and more human-friendly environments. But as the researchers at the Harvard Law School explain, there are many uncharted territories ahead of us.
    To get an idea of where it might go, consider the increasing importance of robotics and automation in our lives. If you’ve ever been in the car or office, or even the house, then you’re familiar with automation. With the right sensors, it’s possible to deploy robots in homes and factories to help with work tasks, from carrying groceries to washing cars. But with the right decisions


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or role]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new experiences and learning opportunities. What's your favorite hobby or activity? I'm always looking for new experiences and learning opportunities. What's your favorite hobby or activity? I'm always looking for new experiences and learning opportunities. What's your favorite hobby or activity? I'm always looking for new experiences and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also a cultural and economic center with a rich history dating back to the Middle Ages. Paris is a popular tourist destination and a major hub for business and finance. The city is home to many famous museums, including the Louvre and the Musée d'Orsay. It is also known for its cuisine, including French cuisine, and is a popular tourist destination for food lovers. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is also home to many international
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more robust AI systems that are designed to be fair
    


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
    Generated text:  [insert fictional character's name]. I'm a professional software developer with a passion for creating user-friendly and efficient software solutions. I'm always looking to improve my skills and stay up-to-date with the latest trends in the industry. I thrive on collaboration and teamwork, and I'm excited to bring my creativity and problem-solving skills to any project I'm involved in. I value hard work and a positive attitude, and I'm always looking for ways to make the world a better place. Thanks for having me! What's your favorite hobby or activity to do in your free time? As a software developer, I find that there are countless creative
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    **Q.** What are the inhabitants called in Paris?
    
    **A.** Parisians are called French. 
    
    **Q.** Where does Paris stand? 
    
    **A.** The capital of France is located on the Mediterranean coast.
    
    **Q.** What is Paris famous for?
    
    **A.** Paris is the most famous city in France, home to the Eiffel Tower, the Louvre Museum, Notre Dame Cathedral, Champs-Elysée, and much more.
    
    **Q.** What is Paris’s climate like?
    
    **A.** The climate in Paris is mild, wet, and rainy.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, and there are several potential trends that could shape the field in the coming years. Here are some of the most promising possibilities:
    
    1. Increased integration with human decision-making: AI is becoming more integrated into decision-making processes, allowing humans to leverage AI's strengths while also minimizing the potential for biases or errors. This could lead to more efficient and effective decision-making in many areas.
    
    2. Better safety and ethical considerations: AI systems are becoming more complex and sophisticated, and there are increasing concerns about their potential to cause harm or misdirection. As AI evolves, there will be more emphasis on safety and ethical considerations, such as preventing AI


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

     I

    'm

     an

     experienced

     [

    job

     title

     or

     hobby

    ]

     who

     is

     passionate

     about

     [

    mot

    ivation

     or

     purpose

    ].

     Here

     are

     some

     key

     points

     about

     me

    :
    


    -

     I

     have

     a

     passion

     for

     [

    job

     title

     or

     hobby

    ]

     that

     drives

     me

     to

     [

    mot

    ivation

     or

     purpose

    ].


    -

     I

     am

     a

     seasoned

     [

    job

     title

     or

     hobby

    ]

     with

     over

     [

    number

    ]

     years

     of

     experience

    .


    -

     I

     am

     dedicated

     to

     [

    mot

    ivation

     or

     purpose

    ]

     and

     always

     strive

     to

     improve

     my

     skills

    .


    -

     I

     am

     confident

     in

     my

     ability

     to

     achieve

     [

    mot

    ivation

     or

     purpose

    ]

     and

     will

     do

     my

     best

    .


    -

     I

     am

     a

     dedicated

     [

    job

     title

     or

     hobby

    ]

     who

     believes

     in

     [

    
    
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

     renowned

     for

     its

     vibrant

     culture

    ,

     historical

     landmarks

    ,

     and

     stunning

     architecture

    ,

     making

     it

     a

     UNESCO

     World

     Heritage

     site

    .

     The

     city

    's

     annual

     E

    iff

    el

     Tower

     parade

    ,

     concerts

    ,

     and

     world

    -ren

    owned

     museums

     attract

     millions

     of

     visitors

     annually

    ,

     making

     it

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    .

     With

     its

     diverse

     cuisine

    ,

     vibrant

     nightlife

    ,

     and

     historic

     significance

    ,

     Paris

     is

     an

     essential

     city

     for

     anyone

     interested

     in

     exploring

     the

     French

     Riv

    iera

    ,

     the

     Lou

    vre

    ,

     and

     much

     more

    .

     It

    's

     a

     must

    -

    see

     destination

     for

     Paris

    ians

     and

     tourists

     alike

    .

     
    


    It

    's

     important

     to

     note

     that

     the

     exact

     date

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     quite

     interesting

     and

     transformative

    .

     Some

     of

     the

     most

     significant

     trends

     include

    :
    


    1

    .

     Machine

     learning

     and

     deep

     learning

    :

     These

     are

     the

     two

     main

     areas

     of

     AI

     that

     are

     expected

     to

     dominate

     the

     future

    .

     Machine

     learning

     will

     allow

     AI

     to

     learn

     from

     data

     and

     make

     decisions

    ,

     while

     deep

     learning

     will

     enable

     machines

     to

     process

     and

     analyze

     complex

     information

    .
    


    2

    .

     Explain

    ability

     and

     transparency

    :

     AI

     systems

     are

     becoming

     more

     and

     more

     sophisticated

    ,

     and

     the

     question

     of

     how

     they

     work

     has

     become

     more

     complex

    .

     As

     researchers

     and

     developers

     work

     to

     make

     AI

     systems

     more

     transparent

    ,

     explain

    able

    ,

     and

     understandable

    ,

     they

     will

     become

     more

     widely

     accepted

    .
    


    3

    .

     AI

     ethics

     and

     privacy

    :

     The

     development

    



```python
llm.shutdown()
```
