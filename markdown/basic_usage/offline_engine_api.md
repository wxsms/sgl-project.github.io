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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.07it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.07it/s]


    2026-05-04 12:23:01,133 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 12:23:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.93it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.20it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.20it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.20it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.20it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.20it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.20it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.20it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.20it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.20it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.20it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:04<00:01, 15.20it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.52it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.52it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.52it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.52it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.52it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.52it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.52it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.52it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.52it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 23.52it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 23.52it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 32.88it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 32.88it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 32.88it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 32.88it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 32.88it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 32.88it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 32.88it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 32.88it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 32.88it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 32.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=44.83 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.80 GB):   3%|▎         | 2/58 [00:00<00:02, 19.49it/s]Capturing num tokens (num_tokens=7168 avail_mem=44.79 GB):   3%|▎         | 2/58 [00:00<00:02, 19.49it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.79 GB):   3%|▎         | 2/58 [00:00<00:02, 19.49it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.79 GB):   3%|▎         | 2/58 [00:00<00:02, 19.49it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.79 GB):   9%|▊         | 5/58 [00:00<00:02, 22.71it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.79 GB):   9%|▊         | 5/58 [00:00<00:02, 22.71it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.78 GB):   9%|▊         | 5/58 [00:00<00:02, 22.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.77 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.49it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.77 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=44.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.49it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.49it/s]Capturing num tokens (num_tokens=3072 avail_mem=44.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.49it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=44.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=44.76 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.75 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.75 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.74 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.74 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.39it/s]Capturing num tokens (num_tokens=960 avail_mem=44.73 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.39it/s] Capturing num tokens (num_tokens=896 avail_mem=44.73 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.39it/s]

    Capturing num tokens (num_tokens=832 avail_mem=44.73 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.39it/s]Capturing num tokens (num_tokens=832 avail_mem=44.73 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=768 avail_mem=44.72 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=704 avail_mem=44.72 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=640 avail_mem=44.72 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=576 avail_mem=44.72 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=512 avail_mem=44.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=512 avail_mem=44.70 GB):  50%|█████     | 29/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=480 avail_mem=44.65 GB):  50%|█████     | 29/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=448 avail_mem=44.65 GB):  50%|█████     | 29/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=416 avail_mem=44.65 GB):  50%|█████     | 29/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=384 avail_mem=44.65 GB):  50%|█████     | 29/58 [00:00<00:00, 43.49it/s]

    Capturing num tokens (num_tokens=352 avail_mem=44.64 GB):  50%|█████     | 29/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=352 avail_mem=44.64 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=320 avail_mem=44.63 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=288 avail_mem=44.63 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=256 avail_mem=44.63 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=240 avail_mem=44.63 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=224 avail_mem=44.62 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=224 avail_mem=44.62 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=208 avail_mem=44.62 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.55it/s]Capturing num tokens (num_tokens=192 avail_mem=43.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.55it/s]

    Capturing num tokens (num_tokens=176 avail_mem=43.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=160 avail_mem=43.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=144 avail_mem=43.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.55it/s]Capturing num tokens (num_tokens=144 avail_mem=43.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.30it/s]Capturing num tokens (num_tokens=128 avail_mem=43.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.30it/s]Capturing num tokens (num_tokens=112 avail_mem=43.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.30it/s]

    Capturing num tokens (num_tokens=96 avail_mem=43.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.30it/s] Capturing num tokens (num_tokens=80 avail_mem=43.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.30it/s]Capturing num tokens (num_tokens=64 avail_mem=43.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.30it/s]Capturing num tokens (num_tokens=64 avail_mem=43.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=48 avail_mem=40.29 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=32 avail_mem=40.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=28 avail_mem=40.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=24 avail_mem=40.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=20 avail_mem=40.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=20 avail_mem=40.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.19it/s]Capturing num tokens (num_tokens=16 avail_mem=40.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.19it/s]Capturing num tokens (num_tokens=12 avail_mem=40.17 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.19it/s]

    Capturing num tokens (num_tokens=8 avail_mem=40.17 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.19it/s] Capturing num tokens (num_tokens=4 avail_mem=40.16 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.19it/s]Capturing num tokens (num_tokens=4 avail_mem=40.16 GB): 100%|██████████| 58/58 [00:01<00:00, 36.41it/s]


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
    Generated text:  Vincent and I am a software developer and coder. I have been programming since I was 14 years old and have developed a passion for programming since then. My skills include JavaScript, HTML/CSS, Python, and Ruby.
    
    My main projects include:
    
    - A web application that allows users to post articles about their experiences on the internet, with comments and links to other relevant articles.
    - A mobile application that provides an API for users to post, comment, and follow articles.
    
    I have a passion for creating clean, simple code and I am always looking for new ways to improve my skills.
    
    How can I get started with coding? What
    ===============================
    Prompt: The president of the United States is
    Generated text:  the leader of the executive branch of the federal government of the United States. The president is the commander-in-chief of the United States Armed Forces and is responsible for guiding the actions of the United States Congress in the performance of their duties.
    Choose your answer: Is the following statement true?
    
    "The President of the United States serves as the head of state, as well as the head of government."
    
    OPTIONS: -yes; -no; Yes, the statement is true. The President of the United States serves as the head of state, as well as the head of government. This is true because the position of president is primarily ceremonial and ceremonial,
    ===============================
    Prompt: The capital of France is
    Generated text:  _____. (Answer: C)
    A. Paris
    B. Brussels
    C. Strasbourg
    D. Lille
    
    To determine the capital of France, let's break down the options:
    
    A. Paris - This is indeed the capital of France.
    
    B. Brussels - This is not the capital of France.
    
    C. Strasbourg - This is not the capital of France.
    
    D. Lille - This is not the capital of France.
    
    Therefore, the correct answer is \boxed{A}. Paris is the capital of France.
    ===============================
    Prompt: The future of AI is
    Generated text:  a real possibility, with a number of organizations, researchers, and individuals continuing to investigate and develop various applications of AI. AI has the potential to revolutionize many industries, from healthcare to finance to education, and it has the potential to have a positive impact on the environment. However, it is important to remember that AI is not perfect and that there are many challenges and ethical concerns that need to be addressed. This is why it is important to continue to explore and learn about AI, and to work towards creating a more ethical and responsible use of this technology. By doing so, we can ensure that AI is used to benefit humanity, not


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a cultural and economic center with a rich history dating back to the Roman Empire and the French Revolution. It is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its diverse cuisine, including French cuisine, and its vibrant nightlife. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. Its status as the capital of France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation: AI is already being used in a wide range of industries, from manufacturing to healthcare to transportation. As automation continues to advance, we can expect to see even more widespread use of AI in various sectors.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need for privacy and security. We will need to ensure that AI systems are designed and
    


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
    Generated text:  [insert name], and I'm a [insert age, gender, occupation, etc.]. I recently graduated from [insert university, school, etc.], and I'm currently working as a [insert profession]. In my free time, I enjoy [insert interests, hobbies, etc.]. I'm also a [insert interest, like music, sports, etc.] and I enjoy [insert interests, like reading, writing, etc.]. I have [insert number of friends, hobbies, etc.], and I like [insert hobbies, like playing board games, hiking, etc.]. And I'm [insert personality traits, like
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the western region of the country, and is the largest and most populous city in the world by population. It is also the capital of the departments of Paris, which include 154 cities, and the regional capital of the Île-de-France metropolitan region. Paris is the seat of the French government, the National Assembly, the Council of State, and the Elysée Palace. It is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and the Musée d'Orsay. The city also has a rich history and has been
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a number of key trends and developments that will shape the direction of progress and impact across the industry. Here are some possible future trends in artificial intelligence:
    
    1. Increased automation and artificial general intelligence: One of the key trends in AI is the increasing automation of routine tasks, with machines able to perform many functions that humans can do with minimal effort. This automation will allow AI to be more efficient and productive, freeing up human resources for more complex and creative tasks.
    
    2. Integration with other technologies: AI will continue to be integrated with other technologies such as the Internet of Things (IoT), blockchain, and quantum computing


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

    'm

     a

     [

    job

     title

    ]

     with

     [

    years

     of

     experience

    ].

     I

     have

     always

     been

     passionate

     about

     [

    specific

     field

     or

     hobby

    ]

     and

     have

     always

     been

     driven

     by

     my

     [

    strength

     or

     goal

    ],

     which

     I

     strive

     to

     achieve

    .

     I

     have

     a

     keen

     sense

     of

     empathy

     and

     am

     always

     looking

     for

     ways

     to

     help

     others

    .

     I

     am

     a

     reliable

     and

     hard

    working

     person

    ,

     and

     I

     am

     always

     ready

     to

     lend

     a

     helping

     hand

     when

     needed

    .

     I

     am

     confident

     in

     my

     abilities

     and

     enjoy

     the

     challenge

     of

     learning

     and

     growing

     in

     my

     field

    .

     I

     am

     excited

     to

     help

     someone

     who

     is

     looking

     for

     a

     [

    career

     or

     role

    ].

     I

     am

     looking

     forward

     to

     learning

     more

     about

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     in

     the

     country

    ,

     with

     a

     population

     of

     over

     

    2

    .

     

    2

     million

     people

    .

     The

     city

     is

     located

     on

     the

     River

     Se

    ine

    ,

     on

     the

     Î

    le

     de

     France

    ,

     and

     is

     the

     second

     largest

     city

     in

     the

     European

     Union

    .

     Paris

     is

     a

     historical

     and

     cultural

     center

    ,

     known

     for

     its

     ancient

     architecture

    ,

     museums

    ,

     and

     art

     galleries

    .

     It

     is

     also

     a

     major

     hub

     for

     finance

     and

     business

    ,

     with

     many

     important

     financial

     institutions

     and

     businesses

     located

     within

     the

     city

    .

     Paris

     is

     also

     home

     to

     many

     famous

     landmarks

     and

     attractions

    ,

     including

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

     It

     is

     a

     bustling

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     different

     trends

    ,

     including

    :
    


     

     

    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     AI

     systems

     are

     likely

     to

     become

     more

     integrated

     with

     human

     intelligence

    ,

     with

     some

     AI

     systems

     being

     able

     to

     operate

     independently

     of

     human

     input,

     while

     others

     rely

     on

     human

     interaction

     to

     function

    .

     This

     may

     result

     in

     more

     natural

     and

     intuitive

     AI

     systems

    ,

     as

     well

     as

     a

     greater

     ability

     to

     understand

     and

     respond

     to

     human

     emotions

     and

     emotions

    .


     

     

    2

    .

     More

     sophisticated

     algorithms

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     they

     are

     likely

     to

     become

     more

     sophisticated

     and

     capable

     of

     making

     more

     accurate

     predictions

     and

     decisions

    .

     This

     may

     lead

     to

     AI

     systems

     becoming

     more

     autonomous

     and

     capable

     of

    



```python
llm.shutdown()
```
