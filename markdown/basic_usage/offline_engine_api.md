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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.50it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.07it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.07it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.38it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.38it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.38it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.50it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.50it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.50it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.50it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.50it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.50it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.50it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.50it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.66 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.66 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.66 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.66 GB):   3%|▎         | 2/58 [00:00<00:02, 18.90it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.66 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.65 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.64 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.64 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.64 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.64 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.64 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.13it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=53.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.62 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.62 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.62 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.61 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.60 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.01it/s]Capturing num tokens (num_tokens=960 avail_mem=53.60 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.01it/s] Capturing num tokens (num_tokens=896 avail_mem=53.60 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.01it/s]

    Capturing num tokens (num_tokens=832 avail_mem=53.59 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.01it/s]Capturing num tokens (num_tokens=832 avail_mem=53.59 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.45it/s]Capturing num tokens (num_tokens=768 avail_mem=53.59 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.45it/s]Capturing num tokens (num_tokens=704 avail_mem=53.59 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.45it/s]Capturing num tokens (num_tokens=640 avail_mem=53.58 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.45it/s]Capturing num tokens (num_tokens=576 avail_mem=53.58 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.45it/s]Capturing num tokens (num_tokens=512 avail_mem=53.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.45it/s]Capturing num tokens (num_tokens=512 avail_mem=53.57 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]Capturing num tokens (num_tokens=480 avail_mem=53.58 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]Capturing num tokens (num_tokens=448 avail_mem=53.58 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]Capturing num tokens (num_tokens=416 avail_mem=53.58 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]Capturing num tokens (num_tokens=384 avail_mem=53.58 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]

    Capturing num tokens (num_tokens=352 avail_mem=53.57 GB):  50%|█████     | 29/58 [00:00<00:00, 43.92it/s]Capturing num tokens (num_tokens=352 avail_mem=53.57 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.66it/s]Capturing num tokens (num_tokens=320 avail_mem=53.57 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.66it/s]Capturing num tokens (num_tokens=288 avail_mem=53.56 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.66it/s]Capturing num tokens (num_tokens=256 avail_mem=53.56 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.66it/s]Capturing num tokens (num_tokens=240 avail_mem=53.56 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.66it/s]Capturing num tokens (num_tokens=224 avail_mem=53.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.66it/s]Capturing num tokens (num_tokens=208 avail_mem=53.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.66it/s]Capturing num tokens (num_tokens=208 avail_mem=53.55 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.23it/s]Capturing num tokens (num_tokens=192 avail_mem=53.55 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.23it/s]Capturing num tokens (num_tokens=176 avail_mem=53.55 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=160 avail_mem=53.54 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.23it/s]

    Capturing num tokens (num_tokens=144 avail_mem=53.54 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=128 avail_mem=53.54 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=128 avail_mem=53.54 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=112 avail_mem=53.54 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=96 avail_mem=53.53 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s] Capturing num tokens (num_tokens=80 avail_mem=53.53 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=64 avail_mem=53.52 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=48 avail_mem=53.52 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=48 avail_mem=53.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=32 avail_mem=53.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=28 avail_mem=53.51 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=24 avail_mem=53.51 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]

    Capturing num tokens (num_tokens=20 avail_mem=53.51 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=16 avail_mem=53.51 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=16 avail_mem=53.51 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.88it/s]Capturing num tokens (num_tokens=12 avail_mem=53.50 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.88it/s]Capturing num tokens (num_tokens=8 avail_mem=53.50 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.88it/s] Capturing num tokens (num_tokens=4 avail_mem=53.50 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.88it/s]Capturing num tokens (num_tokens=4 avail_mem=53.50 GB): 100%|██████████| 58/58 [00:01<00:00, 42.32it/s]


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
    Generated text:  Xiao Yu. I was born in 2000. In the year 2000, I was 15 years old. My friends and I decided to have a farewell party for me. However, I was invited to a new school for the first time by my parents. I was sad because I had not seen my friends there yet. Which of the following would be the most appropriate way to handle Xiao Yu's feelings of sadness? A. Xiao Yu sent a message to his friends at the school inviting them to come together B. Xiao Yu called his mother and told her about the sadness he felt about the school
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 35 years old. 10 years ago, he was x times as old as he was 35 years earlier. What is the value of x?
    To solve this problem, we need to set up an equation based on the information given.
    
    1. Let's denote the current age of the president as \( A \). According to the problem, \( A = 35 \).
    2. Ten years ago, the president's age was \( A - 10 \).
    3. We are given that ten years ago, the president's age was \( x \) times his age 35 years earlier
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. Berlin
    C. London
    D. Moscow
    E. Prague
    Answer:
    
    A
    
    Based on the definition of taxation, the transfer of property rights is a tax ______.
    A. Reliance
    B. Incentive
    C. Expenditure
    D. Redistribution
    Answer:
    
    D
    
    The most significant factor influencing an individual's choice of career is ____
    A. Personal preference
    B. Personal goals
    C. Personal income
    D. Personal interests
    Answer:
    
    D
    
    The purpose of establishing a training program is to train employees to be ______.
    A. Competent
    B. Efficient
    ===============================
    Prompt: The future of AI is
    Generated text:  here. So is the future of data, and the potential for AI to improve the world. AI is not new; it has been around for centuries, and many scientists have been studying it for the better part of a century. But it’s only now that we have the computational power, data, and models needed to apply AI to solve problems and make decisions.
    The most obvious way to use AI in the future is to make decisions about the world around us. With powerful new technologies like machine learning and deep learning, AI can analyze patterns in the data and identify patterns in the world around us. It can also improve the accuracy of medical


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and improve my skills. I'm always eager to learn and grow, and I'm always willing to take on new challenges. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and improve my skills. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a bustling city with a rich cultural heritage and is a major tourist destination. The city is known for its cuisine, fashion, and art, and is a popular destination for tourists from around the world. It is also home to the French Parliament and the French National Museum. Paris is a major tourist destination and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced decision-making. This could lead to more personalized and context-aware AI systems that can better understand and respond to the needs of individuals.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be an increased need for privacy and security measures to protect personal data. This could lead to more advanced privacy-preserving AI techniques and the development of new privacy-preserving algorithms.
    
    3. Greater use of AI in healthcare: AI is already being used in healthcare to
    


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
    Generated text:  [insert first and last name] and I am a [insert profession or occupation] who is passionate about [insert a personal hobby, passion, or interest]. I enjoy [insert something about my hobbies or interests]. If you have any questions or need help, feel free to reach out. I look forward to the possibility of working with you. 🌟👩‍💻✨
    
    ---
    
    **Questions and Responses**
    
    **Q:** How would you describe your personality and what draws you to the fictional world you've created?
    
    **A:** As a fictional character, I embody the essence of a person who is both free and bound by their
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the European Union and one of the most visited cities in the world. The city was founded in 789 AD as the capital of the Carolingian Empire. Paris is known for its art, architecture, food, and music, and it is also home to several iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is often referred to as the "City of Love" due to its romantic and picturesque atmosphere. Paris is the eighth most populous city in the world and the second most populous city in Europe by population. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dynamic and dynamic, and the rate of change is increasing. Here are some possible future trends in AI:
    
    1. Increased efficiency and accuracy: As AI technology continues to improve, we can expect to see significant advancements in the areas of efficiency and accuracy. AI can be used in areas such as manufacturing, finance, and healthcare to increase productivity, reduce errors, and improve decision-making.
    
    2. Greater integration with human decision-making: AI is becoming increasingly integrated into decision-making processes, particularly in areas such as healthcare and finance. However, it is not yet clear how far this integration will go.
    
    3. Greater emphasis on ethical considerations: With


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

    ],

     and

     I

    'm

     a

     young

     professional

     with

     a

     passion

     for

     learning

     new

     things

     and

     pursuing

     my

     dream

     career

    .

     I

     thrive

     on

     collaboration

     and

     communication

    ,

     and

     I

     believe

     that

     great

     teamwork

     can

     drive

     success

    .

     I

    'm

     a

     perfection

    ist

     and

     always

     strive

     for

     excellence

    ,

     even

     if

     it

     means

     pushing

     myself

     to

     the

     limit

    .

     I

     believe

     that

     hard

     work

     and

     dedication

     are

     the

     key

     to

     achieving

     my

     goals

    ,

     and

     I

     strive

     to

     do

     my

     best

     every

     day

    .

     I

    'm

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

     my

     skills

    .

     I

    'm

     confident

     in

     my

     abilities

     and

     I

    'm

     excited

     to

     work

     with

     others

     and

     make

     a

     positive

     impact

     in

     our

     community

    .

     Thank

     you

     for

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     city

     of

     lights

    .

     It

     is

     an

     important

     cultural

     and

     economic

     center

     of

     the

     country

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     music

    ,

     and

     food

    .

     It

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     home

     to

     many

     museums

    ,

     theaters

    ,

     and

     other

     cultural

     institutions

    .

     Paris

     is

     a

     major

     transportation

     hub

     for

     Europe

     and

     plays

     a

     significant

     role

     in

     the

     country

    's

     economy

    .

     As

     of

     

    2

    0

    2

    1

    ,

     the

     population

     of

     Paris

     is

     around

     

    2

    .

    3

     million

    .

     The

     city

     is

     located

     in

     the

     South

     of

     France

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     and

     it

    's

     likely

     that

     we

     will

     see

     a

     continued

     expansion

     of

     its

     applications

     in

     various

     sectors

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     automation

    :

     AI

     will

     likely

     continue

     to

     automate

     many

     of

     the

     tasks

     that

     require

     human

     intelligence

    ,

     such

     as

     data

     analysis

    ,

     decision

    -making

    ,

     and

     problem

    -solving

    .

     This

     will

     lead

     to

     greater

     efficiency

     and

     productivity

     in

     various

     industries

    .
    


    2

    .

     Enhanced

     personal

    ization

    :

     AI

     will

     enable

     businesses

     to

     offer

     more

     personalized

     experiences

     to

     their

     customers

    .

     For

     example

    ,

     AI

     can

     be

     used

     to

     analyze

     customer

     behavior

     and

     preferences

     to

     create

     targeted

     marketing

     campaigns

    .
    


    3

    .

     Improved

     healthcare

    :

     AI

     will

     likely

     play

     a

     key

     role

     in

     improving

     healthcare

     by

    



```python
llm.shutdown()
```
