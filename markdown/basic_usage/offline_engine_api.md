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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.79it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.58s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.58s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:20,  4.58s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:20,  4.58s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]

    Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 12.53it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 12.53it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 12.53it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 12.53it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 12.53it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 12.53it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 12.53it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:02, 12.53it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:01, 17.86it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:01, 17.86it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:01, 17.86it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:01, 17.86it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:01, 17.86it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:01, 17.86it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:01, 17.86it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:01, 17.86it/s]

    Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:05<00:01, 17.86it/s]Compiling num tokens (num_tokens=160):  59%|█████▊    | 34/58 [00:05<00:01, 17.86it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 26.24it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 35.12it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 35.12it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 35.12it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 35.12it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 35.12it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 35.12it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 35.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.27 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.26 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.26 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.26 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.26 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.26 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.25 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.24 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.24 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.23 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.21 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.21 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.19 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s]Capturing num tokens (num_tokens=960 avail_mem=76.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s] Capturing num tokens (num_tokens=896 avail_mem=76.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.98it/s]Capturing num tokens (num_tokens=832 avail_mem=76.20 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=768 avail_mem=76.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=704 avail_mem=76.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=640 avail_mem=76.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=576 avail_mem=76.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=512 avail_mem=76.17 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=512 avail_mem=76.17 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=480 avail_mem=76.19 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=448 avail_mem=76.18 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=416 avail_mem=76.18 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=384 avail_mem=76.18 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.17 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=352 avail_mem=76.17 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=320 avail_mem=76.17 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=288 avail_mem=76.17 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=256 avail_mem=76.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=240 avail_mem=76.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.98it/s]Capturing num tokens (num_tokens=224 avail_mem=76.16 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=224 avail_mem=76.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=208 avail_mem=76.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=192 avail_mem=76.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=176 avail_mem=76.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=160 avail_mem=76.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.05it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=144 avail_mem=76.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=128 avail_mem=76.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=112 avail_mem=76.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=96 avail_mem=76.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s] Capturing num tokens (num_tokens=80 avail_mem=76.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=64 avail_mem=76.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=64 avail_mem=76.13 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=48 avail_mem=76.12 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=32 avail_mem=76.12 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.99it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.12 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=24 avail_mem=76.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=20 avail_mem=76.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=20 avail_mem=76.11 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=16 avail_mem=76.11 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=12 avail_mem=76.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=8 avail_mem=76.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.32it/s] Capturing num tokens (num_tokens=4 avail_mem=76.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=4 avail_mem=76.10 GB): 100%|██████████| 58/58 [00:01<00:00, 39.79it/s]


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
    Generated text:  Evan and I am a computer scientist and software engineer. My main area of expertise is designing and building software systems, primarily for web applications. My past work includes working on a variety of web and mobile applications, as well as developing data management systems. I have been working in the field of web and mobile development since 2006. My academic background includes a Bachelor of Science degree in Computer Science and a Master of Science in Systems Engineering. Currently, I am pursuing a Master of Science in Systems Engineering at MIT.
    I have been involved in many different projects and have had the opportunity to work with a wide range of individuals and organizations
    ===============================
    Prompt: The president of the United States is
    Generated text:  36 years old today. In 10 years, the president's age will be 2/3 times as old as he was 10 years ago. How old is the president today?
    To determine the president's current age, we need to set up an equation based on the information given. Let's denote the president's current age as \( x \).
    
    According to the problem, in 10 years, the president's age will be 2/3 times as old as he was 10 years ago. This can be expressed with the following equation:
    
    \[ x + 10 = \frac{
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Madrid C. London D. Rome
    Answer:
    A
    
    Among the following options, which one represents a mixture?
    A. Water
    B. Air
    C. Carbon monoxide
    D. Ammonia
    Answer:
    B
    
    Which of the following statements is correct? (1) In a chemical reaction, the reactant that is not converted into a product is called a reactant.
    A. Correct
    B. Incorrect
    C. Doesn't matter
    D. Insufficient information
    
    Answer:
    B
    
    Please select the sentence that is grammatically correct and logically sound from the following options. ____
    ===============================
    Prompt: The future of AI is
    Generated text:  at the heart of the smart city initiative from the Department of Defense. The Department of Defense, with its goal of increasing the capabilities and efficiency of the U.S. military, has recently announced plans to develop and deploy AI technologies to support its mission and enhance its operational effectiveness. The initiative aims to improve the situational awareness of its military forces, and to optimize the use of existing resources for mission-critical operations.
    One of the key challenges that the Department of Defense faces is ensuring that the AI technologies developed and deployed are secure. The AI systems are vulnerable to cyberattacks, and the government wants to ensure that these attacks are minimized. To


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major tourist destination and a major economic and financial center in Europe. The city is known for its fashion, art, and cuisine, and is home to many famous museums and galleries. Paris is a vibrant and dynamic city with a diverse population and a strong sense of community. Its status
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could emerge in the coming years:
    
    1. Increased integration of AI into everyday life: As AI becomes more integrated into our daily lives, we are likely to see more widespread adoption of AI technologies. This could include things like voice assistants, self-driving cars, and smart home devices.
    
    2. Greater emphasis on ethical AI: As AI becomes more advanced, there will be a growing emphasis on ethical considerations. This could include things like ensuring that AI systems are transparent, accountable, and fair
    


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
    Generated text:  [Name], and I am [Age] years old. I am an accomplished [occupation] with a passion for [interest or hobby]. I am always looking for ways to [describe a new skill or challenge], and I believe that my enthusiasm for [occupational field] and my ability to work with others are my greatest strengths. I have always been a hardworking and dedicated individual, and I am always eager to learn and grow in my career. I believe in my ability to make a difference and to help others, and I am always striving to contribute to the world in the best way I can. Thank you for considering me for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the south of the country.
    You are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer. Explain in French the meaning of the terms I’m afraid I don’t fully understand. Could you please specify the city I’m referring to? The city I’m thinking of is likely Nice, located in the Côte d’Azur region of southern France, near the Mediterranean Sea.
    Sorry, I don't know which city you're referring to. Can you please specify which city you're talking about? The city you are referring to is Nice, located in the Côte
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dynamic and unpredictable, and it is difficult to predict with certainty what the next wave of innovations will be. However, there are several possible trends that could shape the development of AI in the years to come:
    
    1. Advancements in machine learning and deep learning: Machine learning and deep learning are two areas of AI that are expected to grow and become more advanced over the next few years. These technologies are capable of learning from vast amounts of data and can automate tasks that would previously require human intelligence.
    
    2. Increased reliance on AI in various industries: AI is expected to play a more significant role in various industries in the coming years, including


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

     fictional

     character

    's

     name

    ].

     I

     am

     a

     [

    insert

     fictional

     character

    's

     age

    ,

     gender

    ,

     occupation

    ,

     or

     any

     other

     relevant

     information

     relevant

     to

     the

     character

    ].

     I

     am

     [

    insert

     fictional

     character

    's

     profession

     or

     hobby

    ].

     I

     am

     passionate

     about

     [

    insert

     a

     specific

     hobby

     or

     passion

     of

     the

     character

    ].

     I

     have

     [

    insert

     any

     skills

     or

     abilities

     relevant

     to

     the

     character

    ].

     I

     enjoy

     [

    insert

     any

     hobbies

     or

     interests

     related

     to

     the

     character

    ].

     I

     am

     [

    insert

     any

     personal

     details

     or

     quir

    ks

     of

     the

     character

    ].

     I

     am

     [

    insert

     any

     unusual

     or

     quirky

     characteristics

     of

     the

     character

    ].

     I

     have

     [

    insert

     any

     notable

     achievements

     or

     accomplishments

     of

     the

     character

    ].

     I

     am

     [

    insert

     any

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

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

    .

     Additionally

    ,

     the

     city

     has

     a

     rich

     history

     and

     culture

    ,

     with

     its

     historic

     center

     and

     many

     historic

     districts

     and

     neighborhoods

    .

     Paris

     is

     also

     a

     cosm

    opolitan

     city

     with

     a

     diverse

     population

     and

     many

     cultural

     institutions

    ,

     including

     the

     Museum

     of

     Modern

     Art

     and

     the

     V

    ieux

    -

    Paris

     neighborhood

    .

     Its

     climate

     is

     warm

     and

     humid

    ,

     with

     a

     mild

     winter

     and

     a

     hot

     summer

    ,

     and

     it

     is

     an

     important

     center

     for

     trade

     and

     commerce

    ,

     hosting

     numerous

     events

     and

     festivals

    .

     In

     terms

     of

     tourism

    ,

     Paris

     is

     a

     major

     tourist

     destination

     with

     a

     large

     number

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     dynamic

     and

     is

     influenced

     by

     a

     variety

     of

     factors

     including

     technological

     advancements

    ,

     societal

     shifts

    ,

     and

     changes

     in

     business

     practices

    .

     Some

     possible

     trends

     that

     AI

     is

     likely

     to

     experience

     in

     the

     coming

     years

     include

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     everyday

     life

    :

     AI

     is

     becoming

     increasingly

     integrated

     into

     our

     daily

     lives

    ,

     from

     our

     devices

     and

     vehicles

     to

     healthcare

     and

     finance

    .

     This

     integration

     will

     likely

     continue

     to

     expand

    ,

     with

     more

     applications

     of

     AI

     being

     developed

     and

     adopted

     by

     consumers

    .
    


    2

    .

     Automation

     of

     routine

     tasks

    :

     AI

     is

     likely

     to

     automate

     a

     wide

     range

     of

     tasks

    ,

     from

     manufacturing

     and

     logistics

     to

     customer

     service

     and

     customer

     support

    .

     This

     automation

     will

     likely

     lead

     to

     increased

     efficiency

     and

     lower

     costs

    



```python
llm.shutdown()
```
