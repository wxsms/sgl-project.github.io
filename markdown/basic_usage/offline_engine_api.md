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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.59it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.59it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.59it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.59it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.59it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.59it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.59it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.59it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.59it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.59it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.59it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 17.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 17.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 17.00it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=75.13 GB):   7%|▋         | 4/58 [00:00<00:03, 16.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   7%|▋         | 4/58 [00:00<00:03, 16.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.79 GB):   7%|▋         | 4/58 [00:00<00:03, 16.63it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   7%|▋         | 4/58 [00:00<00:03, 16.63it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.86it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.86it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.86it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.68it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.68it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.94it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.65 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.85it/s] Capturing num tokens (num_tokens=896 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.21it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.21it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.21it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.21it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.21it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.21it/s]Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=384 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.17it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.17it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.15it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.15it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.22it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.22it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 36.63it/s]


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
    Generated text:  Richard and I'm a freelancer. I have experience in graphic design and web development.
    I have been developing software for 2 years now. It's my passion and main area of expertise.
    I'm not a designer, but a graphic designer.
    I have experience in video editing for 2 years. I have experience in video editing for both old and new technologies.
    I have experience in web design for 3 years. I have experience in web design for both old and new technologies.
    I have experience in UI/UX design for 2 years. I have experience in UI/UX design for both old and new technologies.
    I have experience
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military personnel to have on active duty. He believes that the number of personnel should be an even number. The cost of maintaining a military personnel is $200,000 per person per year. If the president has $1,000,000 to spend, what is the maximum number of military personnel that can be supported? To determine the maximum number of military personnel the president can support, we need to divide the total budget by the cost per personnel. The total budget is $1,000,000, and each military personnel costs $200,
    ===============================
    Prompt: The capital of France is
    Generated text: : ____
    A. Paris
    B. Nice
    C. Brest
    D. Var
    Answer:
    
    A
    
    What is the standard length of a fiber optic cable in China?
    A. 150 meters
    B. 250 meters
    C. 500 meters
    D. 1000 meters
    Answer:
    
    C
    
    The function of a data center is to provide a specialized environment for storing and processing data. Which of the following is NOT a function of a data center?
    A. Data storage
    B. Network connectivity
    C. Data processing
    D. Information analysis
    Answer:
    
    B
    ===============================
    Prompt: The future of AI is
    Generated text:  already here
    
    AI is constantly evolving and growing faster than we can keep up. This is why it is crucial to ensure that AI is safe, ethical, and aligned with the values of society.
    
    By supporting AI companies and institutions, we are playing a crucial role in the development of AI and ensuring its widespread adoption.
    
    The world is full of fascinating and groundbreaking developments that will shape our future. While many of these developments are still in the early stages of development, they are already making significant progress. With the passage of time and continued innovation, we are likely to see more and more amazing and transformative technologies.
    
    One of the most promising areas of


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name]: Hi, I'm [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name]: Hi, I'm [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the Middle Ages. It is a major transportation hub and a major tourist destination. The city is known for its fashion industry, art scene, and cuisine, and is a popular tourist destination for its beautiful architecture and historical sites. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. It is a major economic center
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased integration with other technologies: AI is already being integrated into a wide range of devices and systems, but there is potential for even greater integration with other technologies such as IoT, blockchain, and quantum computing.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there is a risk of privacy and security breaches. There is potential for new technologies to be developed that address these issues, such as blockchain-based privacy-preserving AI.
    
    3. Increased focus on ethical considerations: As AI
    


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
    Generated text:  [Name] and I'm a [Age] year old [Occupation/Role]!
    
    I'm passionate about [Your Hobby/Interest/Strength], and I'm always looking for ways to improve it! 😊
    
    I love [Something I'm Good at], and I enjoy [How I Spend My Time/How I Manage My Time]. 🌟
    
    I'm open to learning and trying new things, and I'm always eager to expand my knowledge and skills. I'm always looking for ways to contribute to the world and make a positive impact. 
    
    I'm not afraid to take risks, and I'm always willing to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To verify this statement, please provide a brief explanation of Paris's significance as a major French city, including its notable landmarks, cultural attractions, and historical events. 
    
    1. **Significance of Paris**: Paris, the French capital, is renowned for its rich history, artistic innovation, and cultural richness. Its art school, the École des Beaux-Arts, has been a center of excellence in Europe since its foundation in 1663. The city's skyline, adorned with towering buildings and monuments, includes the Louvre and Notre-Dame Cathedral. The city is also known for its iconic fashion and gastr
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by several key trends, including:
    
    1. Increased focus on ethical AI: As AI becomes more integrated into various industries, there will be increasing scrutiny of its impact on human lives. This includes concerns about the potential for AI to perpetuate existing inequalities, as well as the ways in which it may lead to new forms of discrimination.
    
    2. Improved privacy and data protection: As more AI systems become integrated into everyday life, there will be increasing pressure to protect user data and privacy. This will require greater efforts to ensure that AI systems are designed to be transparent and accountable, and that they are subject to rigorous testing and oversight


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

    'm

     an

     [

    Age

    ],

     [

    Gender

    ]

     [

    or

     any

     other

     gender

    ].

     I

     currently

     live

     in

     [

    Your

     city

    ,

     country

    ,

     etc

    .

    ].

     I

    'm

     a

     [

    Occup

    ation

    ]

     by

     profession

    ,

     and

     I

     enjoy

     [

    Interesting

     hobby

     or

     interest

    ].

     I

     spend

     my

     days

     [

    What

    's

     your

     usual

     day

    ?

    ]

     and

     [

    What

    's

     your

     day

    time

     activity

    ?

    ].

     What

     is

     your

     favorite

     [

    Food

     or

     drink

    ]

     and

     what

     is

     your

     favorite

     [

    Activity

    ?

    ].

     I

    'm

     always

     ready

     to

     learn

     more

     about

     you

    ,

     and

     to

     share

     my

     personal

     experiences

     and

     thoughts

    .

     How

     about

     you

    ,

     [

    Name

    ]?

     Start

     your

     introduction

     with

     a

     polite

     greeting

    ,

     such

     as

     "

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     and

     the

     capital

     of

     France

    .

     It

     is

     located

     in

     the

     North

     of

     the

     country

     on

     the

     left

     bank

     of

     the

     Se

    ine

     River

    .

     The

     city

     has

     an

     area

     of

     

    9

    0

    8

    .

     

    0

    2

     km

    2

     and

     a

     population

     of

     

    2

    ,

    7

    5

    0

    ,

    0

    0

    0

     as

     of

     

    2

    0

    2

    0

    .

     Its

     official

     language

     is

     French

    .

     The

     city

     is

     well

    -known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     architecture

    ,

     as

     well

     as

     its

     food

     culture

    .

     Paris

     is

     a

     cultural

     and

     political

     center

     in

     Europe

    ,

     and

     it

     is

     the

     largest

     city

     in

     France

     by

     land

     area

    .

     It

     is

     home

     to

     the

     E

    iff

    el

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     factors

    ,

     including

     advances

     in

     computing

     power

     and

     data

     availability

    ,

     changes

     in

     policy

     and

     regulation

    ,

     and

     evolving

     societal

     needs

     and

     goals

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

     Improved

     Explain

    able

     AI

    :

     As

     AI

     systems

     become

     more

     complex

     and

     capable

    ,

     we

     will

     see

     a

     rise

     in

     the

     ability

     to

     explain

     their

     decisions

     and

     actions

     to

     humans

    .

     This

     could

     lead

     to

     greater

     transparency

     and

     trust

     in

     AI

     systems

    ,

     but

     also

     raises

     ethical

     and

     legal

     issues

    .
    


    2

    .

     Personal

    ization

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     we

     may

     see

     a

     greater

     emphasis

     on

     personal

    ization

     and

     customization

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     use

     of

    



```python
llm.shutdown()
```
