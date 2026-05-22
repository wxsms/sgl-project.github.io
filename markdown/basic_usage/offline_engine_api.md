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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.77it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:00,  4.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:00,  4.21s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:00,  4.21s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:00,  4.21s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:00,  4.21s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.54it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.59it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.24it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.40it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.64it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.64it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.64it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.64it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.64it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.64it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.64it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.64it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.64it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.64it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.64it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.68it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.68it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.68it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.68it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.68it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.68it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.68it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.81it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=51.23 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=51.23 GB):   3%|▎         | 2/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=51.23 GB):   3%|▎         | 2/58 [00:00<00:03, 16.68it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=51.23 GB):   3%|▎         | 2/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=51.23 GB):   7%|▋         | 4/58 [00:00<00:03, 14.56it/s]Capturing num tokens (num_tokens=6144 avail_mem=51.20 GB):   7%|▋         | 4/58 [00:00<00:03, 14.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=51.19 GB):   7%|▋         | 4/58 [00:00<00:03, 14.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=51.18 GB):   7%|▋         | 4/58 [00:00<00:03, 14.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=51.18 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=51.18 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=51.18 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.87it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=51.17 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=51.17 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=51.17 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.94it/s]Capturing num tokens (num_tokens=3328 avail_mem=51.16 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.94it/s]Capturing num tokens (num_tokens=3072 avail_mem=51.16 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=51.16 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=51.15 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=51.15 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=51.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=51.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=51.15 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.11it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=51.14 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=51.14 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=51.12 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=51.12 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=960 avail_mem=51.14 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.48it/s] Capturing num tokens (num_tokens=896 avail_mem=51.13 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=832 avail_mem=51.13 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=768 avail_mem=51.13 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=704 avail_mem=51.12 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=704 avail_mem=51.12 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=640 avail_mem=51.12 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=576 avail_mem=51.12 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.62it/s]

    Capturing num tokens (num_tokens=512 avail_mem=51.10 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=480 avail_mem=51.12 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=448 avail_mem=51.12 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=448 avail_mem=51.12 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=416 avail_mem=51.12 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=384 avail_mem=51.11 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=352 avail_mem=51.11 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.08it/s]Capturing num tokens (num_tokens=320 avail_mem=51.10 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.08it/s]Capturing num tokens (num_tokens=288 avail_mem=51.10 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.08it/s]Capturing num tokens (num_tokens=288 avail_mem=51.10 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.93it/s]Capturing num tokens (num_tokens=256 avail_mem=51.10 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.93it/s]Capturing num tokens (num_tokens=240 avail_mem=51.09 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.93it/s]

    Capturing num tokens (num_tokens=224 avail_mem=51.09 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.93it/s]Capturing num tokens (num_tokens=208 avail_mem=51.09 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.93it/s]Capturing num tokens (num_tokens=192 avail_mem=51.09 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.93it/s]Capturing num tokens (num_tokens=192 avail_mem=51.09 GB):  71%|███████   | 41/58 [00:01<00:00, 44.35it/s]Capturing num tokens (num_tokens=176 avail_mem=51.08 GB):  71%|███████   | 41/58 [00:01<00:00, 44.35it/s]Capturing num tokens (num_tokens=160 avail_mem=51.08 GB):  71%|███████   | 41/58 [00:01<00:00, 44.35it/s]Capturing num tokens (num_tokens=144 avail_mem=51.08 GB):  71%|███████   | 41/58 [00:01<00:00, 44.35it/s]Capturing num tokens (num_tokens=128 avail_mem=51.07 GB):  71%|███████   | 41/58 [00:01<00:00, 44.35it/s]Capturing num tokens (num_tokens=112 avail_mem=51.07 GB):  71%|███████   | 41/58 [00:01<00:00, 44.35it/s]Capturing num tokens (num_tokens=112 avail_mem=51.07 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=96 avail_mem=51.07 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.10it/s] Capturing num tokens (num_tokens=80 avail_mem=51.07 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.10it/s]

    Capturing num tokens (num_tokens=64 avail_mem=51.06 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=48 avail_mem=51.06 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=32 avail_mem=51.06 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=32 avail_mem=51.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=28 avail_mem=51.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=24 avail_mem=51.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=20 avail_mem=51.04 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=16 avail_mem=51.04 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=12 avail_mem=51.04 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=12 avail_mem=51.04 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.89it/s]Capturing num tokens (num_tokens=8 avail_mem=51.04 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.89it/s] Capturing num tokens (num_tokens=4 avail_mem=51.03 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.89it/s]

    Capturing num tokens (num_tokens=4 avail_mem=51.03 GB): 100%|██████████| 58/58 [00:01<00:00, 38.02it/s]


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
    Generated text:  Arthur and I am 13 years old. I was born in 1993 and I live in Colorado. I have a long story to tell you about my life. My father is a lawyer and my mother is a school teacher. My father has a great job and he likes to give advice to his students. My mother is very nice and always likes to make sure everyone in the family is happy. I have always been very curious and very interested in learning about the world and the people that live there. I also like to learn and I love reading books. I like to travel and I like to explore places. I
    ===============================
    Prompt: The president of the United States is
    Generated text:  a politician, and the president is the head of the executive branch of the government. Given these facts, what does the head of the executive branch of the government do?
    The answer is the president. The president is the head of the executive branch of the government and is responsible for leading the government in a presidential system of government. While the president is a member of the legislative branch, they are not directly responsible for the implementation and enforcement of laws. The president's primary role is to lead the executive branch, which consists of the executive branch of the federal government, the state legislatures, the military, and the other government branches. The
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. This city has an area of 112 square kilometers. Given the rate of change for the area of a city over time, what is the rate of change of the area of the city in square kilometers per year (km²/yr)? The rate of change of the area of a city is calculated by finding the difference in area divided by the time period. In this case, the rate of change of the area of the city is 112 - 45 = 67 square kilometers per year. The answer is 67.0000000000000
    ===============================
    Prompt: The future of AI is
    Generated text:  something we are seeing from the front lines and most interesting to watch. As scientists and researchers on the front lines of AI see the latest advancements in machine learning, they are rethinking the role that humans play in this world. In the past, AI has been seen as a tool to assist humans in the workplace, but in the future, it’s going to be the next human right.
    The role of the human is going to be more about decision making and decision-making rather than just the vehicle that we use to do our job. While AI has allowed humans to learn and gain new skills, it has also shown that humans have a lot


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill/Ability] who has always been [Positive Traits]. I'm passionate about [What I Love to Do]. I'm [What I Do for a Living]. I'm [What I Do for a Living] and I'm [What I Do for a Living]. I'm [What I Do for a Living] and I'm [What I Do for a Living]. I'm [What I Do for a Living] and I'm [What I Do for a Living]. I'm [What I Do for a Living] and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the country and the largest metropolitan area in Europe. It is located on the Seine River and is the seat of government, administration, and culture for the French Republic. Paris is known for its rich history, art, and cuisine, and is a major tourist destination. It is also home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its fashion industry, with many famous designers and boutiques located in the city. Paris is a vibrant and dynamic city that is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends in AI include:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and preferences.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be increased concerns about privacy and security. There will be efforts to develop more secure and transparent AI systems.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes
    


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
    Generated text:  ________ and I'm a/an ________. Who am I? I'm ________ and I'm from ________. I live ________ and I spend a lot of my free time doing ________. I'm currently going through a difficult period in my life, but I know that I have the strength to overcome this. I'm excited to share my story with you all. Here are a few questions to help me get to know you better:
    
    1. How did you get started in the world of storytelling? What inspired you to start writing?
    2. What was the most challenging part of your journey as a storyteller?
    3. What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France and is located on the Île de la Cité, an island off the coast of the French Riviera. The city is known for its iconic Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and various museums, theaters, and restaurants. Paris is home to many famous landmarks and is a bustling metropolis with a rich history and diverse culture. It is also known as the "City of Light" due to its famous light show and its status as one of the world's most visited cities. Paris has played a significant role in French politics, culture, and economy
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly dynamic, with potential paths and developments that will likely shape the way we live and work in the years to come. Here are some possible trends and developments in AI that are expected to influence this direction:
    
    1. Automation and AI-enhanced labor: One potential trend is the rise of automation and AI-enhanced labor, which will increase efficiency and productivity, allowing workers to focus on higher-value tasks. This could lead to new job roles and careers, such as data scientists, AI engineers, and project managers.
    
    2. AI ethics and regulation: There will be increasing pressure to develop clear guidelines and regulations for AI development, so that it can


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

    ’m

     a

     [

    age

    ],

     [

    gender

    ]

     from

     [

    location

    ],

     and

     I

     have

     always

     been

     passionate

     about

     [

    field

    ].

     I

     love

     to

     travel

     and

     explore

     different

     cultures

    ,

     and

     I

     always

     seek

     out

     new

     experiences

     to

     broaden

     my

     hor

    izons

    .

     I

    ’m

     confident

    ,

     driven

    ,

     and

     outgoing

    ,

     and

     I

     thrive

     on

     learning

     and

     growth

    .

     My

     love

     for

     adventure

     and

     exploration

     is

     contagious

    ,

     and

     I

     love

     sharing

     my

     experiences

     with

     others

    .

     I

    ’m

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     develop

    .

     Looking

     forward

     to

     the

     opportunity

     to

     be

     part

     of

     your

     team

    !

     I

    'm

     excited

     about

     the

     opportunity

     to

     work

     with

     [

    Team

     Name

    ]

     and

     see

     what

     interesting

     projects

     we

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     heart

     of

     the

     French

     Riv

    iera

    ,

     the

     largest

     and

     most

     populous

     metropolitan

     area

     in

     Europe

    .

     It

     is

     the

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     France

    ,

     and

     is

     one

     of

     the

     world

    ’s

     most

     important

     cities

    .

     It

     is

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    ,

     among

     many

     other

     iconic

     landmarks

    .

     Paris

     is

     known

     for

     its

     unique

     blend

     of

     French

     and

     Mediterranean

     styles

    ,

     and

     is

     a

     major

     tourist

     destination

    ,

     particularly

     during

     the

     summer

     months

     when

     the

     city

     is

     adorned

     with

     stunning

     sights

     and

     events

    .

     The

     city

    's

     history

     and

     traditions

     are

     deeply

     ingr

    ained

     in

     its

     identity

    ,

     and

     continue

     to

     be

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     growing

     rapidly

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     development

     of

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     autonomy

    :

     With

     the

     development

     of

     advanced

     algorithms

    ,

     AI

     systems

     will

     become

     more

     capable

     of

     making

     decisions

     and

     actions

     on

     their

     own

    .

     This

     will

     lead

     to

     an

     increase

     in

     autonomy

    ,

     as

     AI

     systems

     will

     be

     able

     to

     adapt

     and

     learn

     from

     their

     environment

    .
    


    2

    .

     More

     personalized

     experiences

    :

     AI

     will

     enable

     the

     creation

     of

     more

     personalized

     experiences

     for

     users

    ,

     as

     AI

     can

     learn

     from

     the

     interactions

     with

     users

     and

     provide

     more

     relevant

     recommendations

     and

     responses

    .
    


    3

    .

     Integration

     with

     human

     beings

    :

     AI

     will

     continue

     to

     become

     more

     integrated

     with

     human

     beings

    ,

     as

     it

     will

    



```python
llm.shutdown()
```
