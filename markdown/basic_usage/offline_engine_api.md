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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.48it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.92it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.92it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]

    Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 25.47it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 25.47it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 25.47it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 25.47it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 25.47it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 25.47it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 25.47it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 25.47it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 25.47it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:05<00:00, 25.47it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 34.55it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 34.55it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 34.55it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 34.55it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 34.55it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 34.55it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.36it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   7%|▋         | 4/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):   7%|▋         | 4/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):   7%|▋         | 4/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.31 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.68it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.09it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.09it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.39it/s] Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.35it/s]

    Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.54it/s]Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.54it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.54it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.54it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.54it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.54it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  62%|██████▏   | 36/58 [00:00<00:00, 44.97it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  62%|██████▏   | 36/58 [00:00<00:00, 44.97it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:00<00:00, 44.97it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.97it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.97it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.97it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  71%|███████   | 41/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  71%|███████   | 41/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.02it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.02it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.02it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.40it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.40it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.40it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.40it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.40it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.40it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.91it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.91it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.91it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 40.04it/s]


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
    Generated text:  Sarah. My first name is Sarah. My last name is Johnson. What is your name? Your name is Sarah. I am a member of the Johnson family. What is your last name? My last name is Johnson. I am a member of the Johnson family. Does your last name start with a letter? No, my last name does not start with a letter. I can’t remember my last name. When you first met me, what did you ask me to do? I asked you what I was. How did you say that? That was my last question. I did not ask you to describe yourself. When you were
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the Vice President. In how many ways can a committee of 3 people be formed from the 248 members of the Senate if it is known that the vice president will be one of the members of the committee? To determine the number of ways to form a committee of 3 people from the 248 members of the Senate where one of the members will be the vice president, we can proceed as follows:
    
    1. **Choose the Vice President**: There are 248 possible choices for the vice president since any one of the 248 members of the Senate can be the vice president.
    
    2
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. As of today, it is home to approximately 2, 000, 000 people. A man has come to Paris with an amount of 100, 000 euros. He spends a certain amount on a stay and a vacation, and he still has 20, 000 euros left. If the amount he spent on a stay is 3 times the amount he spent on a vacation, how much did he spend on a stay?
    
    To determine how much the man spent on a stay, we can set up a system of equations based on the information given.
    
    Let
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but for now, scientists have made significant progress in understanding some of the key challenges. The advancement of technologies like deep learning and other forms of artificial intelligence may lead to the creation of a future where human beings are fully dependent on AI, or even become entirely dependent on it. It's a scary thought, but the truth is that the future of AI has already begun to shape the world we live in today.
    AI is revolutionizing industries across the globe, and it's affecting everything from healthcare to transportation to finance. The advancements in AI have created new job opportunities and has paved the way for future job growth. However, the rise


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your character, such as "a creative problem-solver" or "an expert in [industry]".] I enjoy [insert a short description of your interests, such as "reading", "traveling", or "cooking"]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many world-renowned museums, theaters, and other cultural institutions. Paris is a popular tourist destination and a major economic and financial center in Europe. It is also home to the French Parliament and the French National Library. The city is known for its rich history, diverse culture, and vibrant nightlife. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased automation: As AI technology continues to improve, we are likely to see more automation in various industries, including manufacturing, transportation, and customer service. This will lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI ethics and privacy: As AI technology becomes more advanced, there will be a need to address ethical and privacy concerns. This will require a more transparent and accountable
    


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
    Generated text:  [First Name] and I'm a [Job Title] with [Company Name]. I love [Job Title] because [Short Answer]. I'm a professional who is always [Positive Adjective], [Positive Adjective], and I strive to [Positive Goal]. I believe in [Reason For Success], and I believe in [Reason For Success]. I've always been passionate about [Area of Expertise], and I work to [Achievement]. I'm a [Positive Adjective], and I'm always [Positive Adjective]. I strive to [Achievement]. I'm a [Positive Adjective], and I'm always [Positive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, and it is the largest city and the second most populous city in the European Union. It is also home to many museums, including the Louvre, the most famous of which is the Winged Victory of Samothrace, a Bronze Age Mycenaean sculpture from Greece. The city is also famous for its historical landmarks, including the Eiffel Tower, the Louvre, and the Notre-Dame Cathedral, which is considered to be the most important church in the world. Paris is known for its French cuisine, art, and culture, and it is a popular destination for tourists from all over the world. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of potential and exciting possibilities. Some of the trends that are likely to shape the AI landscape in the coming years include:
    
    1. Increased automation: AI is already becoming more efficient and precise, and we can expect it to continue to advance and improve at a rapid pace. This means that we will see more automation in industries such as manufacturing, transportation, and healthcare.
    
    2. AI will become more integrated with human activities: One of the biggest challenges facing AI is that it will need to work alongside humans to perform tasks effectively. However, there is a growing trend towards AI that will become more integrated with human activities, such as language translation


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

     Alex

    ,

     and

     I

    'm

     a

     self

    -employed

     digital

     marketing

     consultant

    .

     I

     have

     been

     helping

     small

     businesses

     increase

     their

     online

     visibility

     through

     SEO

    ,

     content

     creation

    ,

     and

     outreach

     to

     potential

     customers

    .

     I

     specialize

     in

     creating

     high

    -quality

     content

     that

     reson

    ates

     with

     my

     target

     audience

     and

     provides

     them

     with

     valuable

     insights

     to

     help

     them

     succeed

     in

     their

     businesses

    .

     I

     love

     working

     with

     clients

     to

     find

     the

     best

     approach

     for

     their

     specific

     needs

     and

     goals

    .

     I

    'm

     excited

     to

     start

     a

     new

     chapter

     in

     my

     career

     and

     look

     forward

     to

     seeing

     what

     new

     adventures

     await

     me

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     statement

     succinct

    ly

     captures

     the

     main

     facts

     about

     Paris

    ,

     which

     are

     its

     historical

     significance

    ,

     population

    ,

     and

     cultural

     prominence

    .

     It

     does

     not

     include

     any

     additional

     information

     beyond

     this

     core

     fact

    .

     
    


    I

    'm

     sorry

    ,

     but

     that

     statement

     is

     not

     entirely

     accurate

    .

     While

     Paris

     is

     indeed

     the

     capital

     of

     France

    ,

     it

     is

     not

     the

     largest

     city

     in

     the

     country

    .

     The

     largest

     city

     is

     indeed

     Lyon

    ,

     which

     has

     a

     population

     of

     around

     

    4

    4

    0

    ,

    0

    0

    0

     people

    .

     Another

     notable

     city

     in

     France

     is

     Paris

    ,

     known

     for

     its

     historical

     significance

    ,

     cultural

     attractions

    ,

     and

     vibrant

     nightlife

    .

     While

     Lyon

     has

     a

     smaller

     population

     than

     Paris

    ,

     it

     is

     not

     the

     capital

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     dynamic

     and

     unpredictable

    ,

     and

     there

     are

     many

     potential

     trends

     and

     areas

     of

     focus

     that

     could

     shape

     the

     technology

     in

     the

     coming

     years

    .

     Some

     of

     the

     most

     promising

     areas

     of

     research

     and

     development

     include

    :
    


    1

    .

     Deep

     learning

     and

     machine

     learning

    :

     These

     areas

     of

     AI

     are

     focused

     on

     developing

     more

     powerful

     and

     accurate

     algorithms

     that

     can

     handle

     complex

     tasks

     and

     data

    .

     Researchers

     are

     looking

     at

     how

     to

     improve

     the

     efficiency

     and

     accuracy

     of

     these

     algorithms

    ,

     and

     how

     they

     can

     be

     used

     to

     solve

     more

     challenging

     problems

    .
    


    2

    .

     Natural

     language

     processing

    :

     This

     area

     of

     AI

     is

     focused

     on

     developing

     algorithms

     that

     can

     understand

     and

     interpret

     human

     language

    .

     This

     could

     have

     a

     wide

     range

     of

     applications

    ,

     including

     translation

    ,

    



```python
llm.shutdown()
```
