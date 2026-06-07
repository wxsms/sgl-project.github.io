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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.01it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.01it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.73it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.73it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.73it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.73it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.73it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.73it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.73it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.73it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 18.44it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 18.44it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 18.44it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 18.44it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 18.44it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 18.44it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.44it/s]

    Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 18.44it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 18.44it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 18.44it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 26.46it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 26.46it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 26.46it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 26.46it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 26.46it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 26.46it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 26.46it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 26.46it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 26.46it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 26.46it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 26.46it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 36.28it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 36.28it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 36.28it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 36.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.62 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.61 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.61 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=69.61 GB):   3%|▎         | 2/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.59 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.42it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=69.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.15 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.15 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.13 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.13 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=69.13 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=69.11 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s]

    Capturing num tokens (num_tokens=960 avail_mem=69.12 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s] Capturing num tokens (num_tokens=896 avail_mem=69.12 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=896 avail_mem=69.12 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.22it/s]Capturing num tokens (num_tokens=832 avail_mem=69.12 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.22it/s]Capturing num tokens (num_tokens=768 avail_mem=69.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.22it/s]Capturing num tokens (num_tokens=704 avail_mem=69.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.22it/s]Capturing num tokens (num_tokens=640 avail_mem=69.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.22it/s]Capturing num tokens (num_tokens=576 avail_mem=69.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.22it/s]Capturing num tokens (num_tokens=576 avail_mem=69.11 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=512 avail_mem=69.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=480 avail_mem=69.11 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=448 avail_mem=69.11 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.00it/s]

    Capturing num tokens (num_tokens=416 avail_mem=69.10 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=384 avail_mem=69.10 GB):  48%|████▊     | 28/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=384 avail_mem=69.10 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=352 avail_mem=69.10 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=320 avail_mem=69.09 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=288 avail_mem=69.09 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=256 avail_mem=69.09 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=240 avail_mem=69.08 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=240 avail_mem=69.08 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.03it/s]Capturing num tokens (num_tokens=224 avail_mem=69.08 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.03it/s]Capturing num tokens (num_tokens=208 avail_mem=69.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.03it/s]Capturing num tokens (num_tokens=192 avail_mem=69.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.03it/s]

    Capturing num tokens (num_tokens=176 avail_mem=69.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.03it/s]Capturing num tokens (num_tokens=160 avail_mem=69.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.03it/s]Capturing num tokens (num_tokens=160 avail_mem=69.07 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=144 avail_mem=69.07 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=128 avail_mem=69.06 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=112 avail_mem=69.06 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=96 avail_mem=69.06 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.50it/s] Capturing num tokens (num_tokens=80 avail_mem=69.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=80 avail_mem=69.05 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=64 avail_mem=69.05 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=48 avail_mem=69.04 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=32 avail_mem=69.04 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.02it/s]

    Capturing num tokens (num_tokens=28 avail_mem=69.04 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=24 avail_mem=69.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=24 avail_mem=69.03 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=20 avail_mem=69.03 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=16 avail_mem=69.03 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=12 avail_mem=69.02 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=8 avail_mem=69.02 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.97it/s] Capturing num tokens (num_tokens=4 avail_mem=69.02 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=4 avail_mem=69.02 GB): 100%|██████████| 58/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=4 avail_mem=69.02 GB): 100%|██████████| 58/58 [00:01<00:00, 41.03it/s]


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
    Generated text:  Sarah and I'm in year 10, I have already been working hard on my studies. I want to know the best way to prepare for the SAT? How do I prepare for this?
    Preparing for the SAT can seem overwhelming at first, but with the right strategies, you can improve your chances of success. Here are some steps to help you prepare for the SAT:
    
    1. Understand the test: Before starting to prepare, it's important to understand the format of the SAT. The test consists of four sections: Reading, Writing and Language, Math, and Science. Each section has different types of questions and you can expect to
    ===============================
    Prompt: The president of the United States is
    Generated text:  in the shape of a perfect cube. Its surface area is 150 square units. What is the volume of the president of the United States in cubic units? The surface area of a cube is given by the formula $6s^2$, where $s$ is the length of one side of the cube. In this case, we know that the surface area is 150 square units, so we can set up the equation $6s^2 = 150$ to solve for $s$.
    Dividing both sides by 6, we get $s^2 = 25$. Taking
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A: Paris B: London C: Rome D: Paris and London
    To determine the capital of France, let's consider the options provided:
    
    A: Paris - This is the capital of France.
    B: London - This is the capital of the United Kingdom.
    C: Rome - This is the capital of Italy.
    D: Paris and London - This combination of cities is not the capital of France.
    
    Given these options, the correct capital of France is Paris. Therefore, the correct answer is:
    
    A: Paris. 
    
    Note: The other options (London, Rome) are not capitals of their respective countries. London is the
    ===============================
    Prompt: The future of AI is
    Generated text:  changing. By 2025, an estimated 250 million people worldwide will have at least one AI assistant, and the average user will have a personal assistant with at least 60 million interactions with it. The average user will spend over 50 billion hours a year interacting with AI assistants. The future of AI is moving from one user to many. The future of AI is moving from the user to many. The future of AI is moving from one user to many. What is the future of AI? Let's answer next to each question.
    A: 1. Growing in importance 2. More powerful


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in Europe by population. It is located on the Seine River and is the seat of government for the country. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is also home to many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage. It is also known for its fashion industry, and is home to many famous fashion designers. The city is a major economic center in Europe
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to the same level of complexity and complexity as humans. This could lead to more sophisticated and nuanced AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be an increased need for privacy and security measures to protect the data and personal information that is generated
    


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
    Generated text:  Jane. I am a bright, curious, and adventurous 25-year-old who enjoys exploring the world and learning new things. I love trying new cuisines, reading books, and traveling to new places. I am also a strong communicator and enjoy helping others. I love to think outside the box and try new things to solve problems. What kind of character do you think I am? Jane, a bright and curious 25-year-old who enjoys exploring the world and learning new things, reads books, travels, reads books, and loves to think outside the box to solve problems. Her personality is energetic, curious, and adventurous,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city also hosts numerous museums and art galleries, as well as fashion and food markets. Paris is a cultural hub and a major financial center, with a diverse population of about 1.8 million people. Its reputation as a world-class city attracts tourists from around the world. The city's history, including its role in the French Revolution and Napoleonic Wars, adds to its rich cultural heritage. Paris is a popular tourist destination, with millions of visitors annually. The city's infrastructure is well-maintained
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be very exciting, and there are many potential trends that could shape its development. Here are some of the most likely possibilities:
    
    1. Advanced algorithms and machine learning: As AI technology continues to evolve, we may see the development of even more sophisticated algorithms that can process and analyze large amounts of data more quickly and accurately. This could lead to breakthroughs in areas such as image and speech recognition, natural language processing, and robotics.
    
    2. Increased focus on ethical considerations: As AI systems become more integrated into our daily lives, there will be increasing pressure to address ethical concerns. This could include issues such as bias in algorithms, privacy


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

     a

     software

     engineer

     with

     [

    Company

    ].

     I

     have

     been

     working

     at

     [

    Company

    ]

     for

     [

    Number

    ]

     years

    ,

     and

     I

     specialize

     in

     developing

     and

     implementing

     cutting

    -edge

     software

     solutions

    .

     I

     bring

     a

     unique

     blend

     of

     creativity

     and

     technical

     expertise

     to

     every

     project

     I

     work

     on

    ,

     and

     I

     am

     constantly

     seeking

     out

     new

     ways

     to

     improve

     my

     skills

     and

     stay

     up

    -to

    -date

     with

     the

     latest

     developments

     in

     technology

    .

     I

     am

     always

     looking

     for

     the

     next

     big

     thing

     in

     software

    ,

     and

     I

     believe

     that

     being

     an

     entrepreneur

     is

     a

     great

     way

     to

     connect

     with

     clients

     and

     grow

     my

     career

    .

     Thank

     you

     for

     considering

     my

     introduction

    !

     May

     I

     have

     your

     name

     and

     company

     name

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     in

     the

     south

     of

     the

     country

     and

     on

     the

     island

     of

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     population

    ,

     the

     fourth

     largest

     by

     land

     area

    ,

     and

     the

     fourth

     largest

     by

     water

     area

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     culture

    ,

     and

     vibrant

     nightlife

    ,

     which

     draw

     millions

     of

     tourists

     each

     year

    .

     Paris

     is

     also

     a

     major

     financial

     and

     cultural

     hub

     of

     the

     country

    ,

     with

     many

     world

    -ren

    owned

     museums

    ,

     theaters

    ,

     and

     art

     galleries

    .

     It

     is

     a

     symbol

     of

     French

     identity

     and

     a

     major

     tourist

     destination

    ,

     with

     numerous

     landmarks

    ,

     museums

    ,

     and

     restaurants

     to

     visit

    .

     Despite

     its

     fame

    ,

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     unpredictable

     and

     diverse

    ,

     with

     various

     trends

     expected

     to

     shape

     its

     development

     and

     impact

    .

     Here

     are

     some

     possible

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     integration

     with

     human

     expertise

    :

     As

     AI

     becomes

     more

     advanced

    ,

     we

     are

     likely

     to

     see

     more

     integration

     with

     human

     expertise

     and

     knowledge

    .

     This

     could

     lead

     to

     more

     personalized

     and

     context

    -aware

     AI

     that

     is

     better

     able

     to

     understand

     and

     respond

     to

     human

     needs

     and

     desires

    .
    


    2

    .

     Faster

     and

     more

     efficient

     AI

    :

     AI

     is

     expected

     to

     become

     faster

     and

     more

     efficient

     in

     processing

     and

     analyzing

     large

     amounts

     of

     data

    ,

     which

     could

     lead

     to

     more

     effective

     decision

    -making

     and

     automation

    .
    


    3

    .

     AI

    -driven

     innovations

    :

     AI

     will

     continue

     to

     drive

     new

     innovations

    



```python
llm.shutdown()
```
