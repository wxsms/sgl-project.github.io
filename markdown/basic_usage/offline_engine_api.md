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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:39,  4.90s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:39,  4.90s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:39,  4.90s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:39,  4.90s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:39,  4.90s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.98it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.98it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.98it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.98it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.98it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.98it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.98it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.98it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.98it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.98it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.48it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.48it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.48it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.48it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.48it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.48it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.48it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.48it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.48it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.89it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.89it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.89it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.89it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.89it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.89it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.89it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.89it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.89it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.33it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 14.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 14.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 14.32it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 14.32it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 19.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 19.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 19.76it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 19.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 19.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.45it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.34it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.34it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.34it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.34it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.34it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.71it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.71it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.71it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.71it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.71it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.63it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.63it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.63it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.63it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.63it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.47it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.47it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.47it/s]Capturing num tokens (num_tokens=224 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.47it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.47it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 28.47it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=160 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.62it/s]Capturing num tokens (num_tokens=112 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.62it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.62it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.62it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.62it/s]Capturing num tokens (num_tokens=64 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.99it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.99it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.99it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.99it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.99it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=20 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=16 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.21it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=4 avail_mem=74.24 GB): 100%|██████████| 58/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=4 avail_mem=74.24 GB): 100%|██████████| 58/58 [00:01<00:00, 33.01it/s]


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
    Generated text:  Just, and I am a student at the University of Cape Town, a science and engineering university in South Africa. Currently, I am studying Materials Science and Engineering and I am planning a project that will be based on my interest in the process of foaming, which will involve using foam as a manufacturing material for a product. The main objective of my project is to understand the foaming process of foam and its applications in the manufacturing industry. I am interested in the industrial applications of foams and how they can be used in different industries. Also, I am interested in the use of foams for their energy-saving potential. However, I
    ===============================
    Prompt: The president of the United States is
    Generated text:  the leader of the executive branch of the government, and he is the chief executive of the executive branch. An executive branch, or executive, is the branch of government that executes the laws of the country. The president is chosen by the people for a four-year term. A president is the chief executive of the executive branch, and the president is the head of government. The president must approve all executive orders or executive actions, and must have a majority of the Senate to veto legislation. The president also has the right to issue orders, a power known as the powers of the presidency. The president is sworn in by a state governor and becomes
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Its official language is French. Which of the following statements about Paris is incorrect?
    A. Paris is the capital of France.
    B. Paris is a city in France.
    C. The French national language is French.
    D. Paris is the largest city in France.
    Answer:
    D
    
    The height of a person relative to the ground is 1.7m. The height of a pole relative to the ground is 2.5m. When the person and the pole are at the same height, the gravitational force acting on the person is _______ times that on the pole. 
    A. 3 times
    B.
    ===============================
    Prompt: The future of AI is
    Generated text:  looking increasingly focused on big data and machine learning. Given the importance of big data and machine learning in today’s world, it’s no wonder that many companies and individuals are working to leverage these technologies for their own purposes.
    But what exactly are these technologies, and how do they work? In this blog post, we’ll take a look at the basics of big data and machine learning and how they can be applied to a wide range of industries.
    We’ll cover the different types of data that can be used in these technologies, as well as the different ways that they can be processed and analyzed. We’ll also examine some of the key players


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a city with a rich history and diverse culture. It is located in the south of France and is known for its beautiful architecture, vibrant nightlife, and world-class museums. Paris is a popular tourist destination and a major economic center, with a population of over 2.5 million people. The city is home to many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its cuisine, with a wide variety of delicious dishes and a vibrant food scene. The city is a cultural hub and a major transportation hub,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This will lead to the development of new jobs and the displacement of older, less skilled workers. However, it will also create new opportunities for people who are skilled in AI and robotics.
    
    2. AI ethics and privacy: As AI technology becomes more advanced, there will be a need to address
    


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
    Generated text:  [insert name of the character]. I’m a [insert profession or occupation], and I specialize in [insert skill or area of expertise]. I’ve been working with you for [insert number of years] and I truly believe that my expertise will help you achieve [insert goal or achievement]. Come and see me, and I’ll see you succeed! Can’t wait to hear what you have to say! 🎉✨ #Interpersonal #SkillShare #SelfIntroduction
    Hey there, my name is [insert name of the character]. I’m a [insert profession or occupation] and I specialize in [insert skill or area of expertise
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Fact: Paris is the capital city of France. It is the largest city in the European Union and one of the world's most famous cities. The city is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major center for art, fashion, and music, attracting visitors from around the world. Paris is a popular tourist destination and has played a significant role in French history and culture throughout its history. The city is also known for its wine industry, which is one of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, as the technology is constantly evolving and the implications of its development are yet to be fully understood. However, there are some potential future trends that may occur, including:
    
    1. Self-driving cars: Self-driving cars are expected to become more prevalent in the future as technology advances and the demand for safe, efficient, and reliable transportation systems increases.
    
    2. Personalized AI: The use of AI in personalization is also expected to increase as more data is collected and analyzed to provide more accurate and relevant recommendations to users.
    
    3. AI ethics and regulation: As AI continues to evolve and become more complex, it is likely that the


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

     come

     from

     [

    Country

    ].

     I

    'm

     a

    /an

     [

    Age

    ]

     year

     old

     [

    Gender

    ]

     [

    Name

    ].

     I

    'm

     [

    Occup

    ation

    ].

     I

    'm

     here

     to

     [

    What

     you

    're

     here

     to

     do

     or

     talk

     about

    ].

     I

    'm

     here

     to

     learn

    ,

     grow

    ,

     and

     evolve

    .

     I

    'm

     excited

     to

     meet

     you

    !

     Feel

     free

     to

     ask

     me

     anything

     you

    'd

     like

    .

     I

    'm

     [

    What

     you

    're

     here

     to

     do

     or

     talk

     about

    ].

     [

    What

     you

    're

     here

     to

     do

     or

     talk

     about

    ].


    Certainly

    !

     Here

    's

     a

     neutral

     self

    -int

    roduction

     for

     a

     fictional

     character

     in

     a

     clear

    ,

     concise

    ,

     and

     formal

     tone

    :
    


    ---
    


    Hello

    ,

     my

     name

     is

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     The

     city

     is

     located

     in

     the

     heart

     of

     the

     French

     region

     of

     Î

    le

    -de

    -F

    rance

     and

     is

     home

     to

     the

     country

    ’s

     largest

     city

     and

     the

     second

    -largest

     metropolitan

     area

     in

     the

     world

    .

     Paris

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

     Notre

     Dame

     Basil

    ica

    .

     The

     city

     also

     has

     a

     rich

     history

    ,

     including

     ancient

     ruins

     and

     medieval

     cast

    les

    ,

     and

     is

     home

     to

     many

     prestigious

     institutions

    ,

     including

     the

     French

     National

     Library

     and

     the

     University

     of

     Paris

    .

     Despite

     its

     size

    ,

     Paris

     remains

     one

     of

     the

     most

     cosm

    opolitan

     cities

     in

     Europe

     and

     a

     popular

     tourist

     destination

     for

     millions

     of

     visitors

     annually

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     machine

     learning

    ,

     the

     development

     of

     new

     hardware

     and

     software

     technologies

    ,

     and

     the

     impact

     of

     new

     societal

     and

     environmental

     factors

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     With

     the

     increasing

     amount

     of

     data

     being

     collected

     and

     analyzed

    ,

     the

     ability

     to

     create

     personalized

     experiences

     will

     become

     more

     sophisticated

     and

     efficient

    .

     AI

    -powered

     chat

    bots

    ,

     virtual

     assistants

    ,

     and

     personalized

     learning

     platforms

     will

     become

     more

     effective

     at

     understanding

     user

     needs

     and

     preferences

    .
    


    2

    .

     Greater

     Integration

     with

     Human

     Behavior

    :

     AI

     will

     continue

     to

     integrate

     more

     closely

     with

     human

     behavior

     to

     improve

     decision

    -making

     and

     provide

     more

     accurate

     predictions

     and

     recommendations

    .

     This

    



```python
llm.shutdown()
```
