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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.32it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.68it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.57it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.57it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.57it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.57it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.57it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.57it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.57it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.57it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.00it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.00it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.00it/s]

    Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.00it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.00it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.00it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.00it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.00it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 24.18it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 24.18it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 24.18it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 24.18it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 24.18it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 24.18it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 24.18it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 24.18it/s]

    Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 24.18it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 24.18it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 33.02it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.36 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.36 GB):   3%|▎         | 2/58 [00:00<00:03, 18.41it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.36 GB):   3%|▎         | 2/58 [00:00<00:03, 18.41it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.36 GB):   3%|▎         | 2/58 [00:00<00:03, 18.41it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.36 GB):   3%|▎         | 2/58 [00:00<00:03, 18.41it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.36 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.35 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.34 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.34 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.34 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=57.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.32 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.32 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.31 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.31 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.31 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.30 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.30 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.30it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.30it/s]Capturing num tokens (num_tokens=960 avail_mem=57.30 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.30it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=57.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.30it/s]Capturing num tokens (num_tokens=832 avail_mem=57.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.30it/s]Capturing num tokens (num_tokens=832 avail_mem=57.29 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=768 avail_mem=57.29 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=704 avail_mem=57.28 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=640 avail_mem=57.28 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=576 avail_mem=57.28 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=512 avail_mem=57.26 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.95it/s]Capturing num tokens (num_tokens=512 avail_mem=57.26 GB):  50%|█████     | 29/58 [00:00<00:00, 43.48it/s]Capturing num tokens (num_tokens=480 avail_mem=57.28 GB):  50%|█████     | 29/58 [00:00<00:00, 43.48it/s]Capturing num tokens (num_tokens=448 avail_mem=57.28 GB):  50%|█████     | 29/58 [00:00<00:00, 43.48it/s]Capturing num tokens (num_tokens=416 avail_mem=57.28 GB):  50%|█████     | 29/58 [00:00<00:00, 43.48it/s]

    Capturing num tokens (num_tokens=384 avail_mem=57.27 GB):  50%|█████     | 29/58 [00:00<00:00, 43.48it/s]Capturing num tokens (num_tokens=352 avail_mem=57.27 GB):  50%|█████     | 29/58 [00:00<00:00, 43.48it/s]Capturing num tokens (num_tokens=352 avail_mem=57.27 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.26it/s]Capturing num tokens (num_tokens=320 avail_mem=57.26 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.26it/s]Capturing num tokens (num_tokens=288 avail_mem=57.26 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.26it/s]Capturing num tokens (num_tokens=256 avail_mem=57.26 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.26it/s]Capturing num tokens (num_tokens=240 avail_mem=57.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.26it/s]Capturing num tokens (num_tokens=224 avail_mem=57.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.26it/s]Capturing num tokens (num_tokens=224 avail_mem=57.25 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.58it/s]Capturing num tokens (num_tokens=208 avail_mem=57.25 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.58it/s]Capturing num tokens (num_tokens=192 avail_mem=57.25 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=176 avail_mem=57.24 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.58it/s]

    Capturing num tokens (num_tokens=160 avail_mem=57.24 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=144 avail_mem=57.24 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=144 avail_mem=57.24 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=128 avail_mem=57.24 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=112 avail_mem=57.23 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=96 avail_mem=57.23 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s] Capturing num tokens (num_tokens=80 avail_mem=57.23 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=64 avail_mem=57.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=64 avail_mem=57.22 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=48 avail_mem=57.22 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=32 avail_mem=57.22 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=28 avail_mem=57.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]

    Capturing num tokens (num_tokens=24 avail_mem=57.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=20 avail_mem=57.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.23it/s]Capturing num tokens (num_tokens=20 avail_mem=57.20 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=16 avail_mem=57.20 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=12 avail_mem=57.20 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=8 avail_mem=57.20 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.57it/s] Capturing num tokens (num_tokens=4 avail_mem=57.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.57it/s]Capturing num tokens (num_tokens=4 avail_mem=57.19 GB): 100%|██████████| 58/58 [00:01<00:00, 41.77it/s]


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
    Generated text:  Sarah. I'm a five-year-old girl. I can speak English and some other languages. I like to help others. When I'm not in the school, I go to my grandparents' house. They live in a small town. They have many children. They have two sons. I like to draw. I have a lot of pictures. I like to draw animals, birds, and flowers. My favorite color is pink. The flowers in my pictures are pink. My grandmother is a nice lady. She gives me lots of lovely presents. I like her. I'm very happy. Can you tell me about your family? What
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to estimate the average amount of money each person saves each month. In order to do this, the president needs to know how many different values lie within the interval from $500 to $600. What is the minimum number of people the president needs to interview? To determine the minimum number of people the president needs to interview, we need to find the smallest integer \( n \) such that the number of values of \( n \) that lie within the interval from 500 to 600 is at least 100. This is because we can either have 0 values within the interval
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Nice
    C. London
    D. Moscow
    Answer: A
    
    The capital of Greece is ______.
    A. Athens
    B. Rome
    C. Athens
    D. Athens
    Answer: A
    
    When the total investment of an enterprise reaches ____ million yuan, it should be reported to the local government's work safety supervision department for record. 
    A. 500
    B. 1000
    C. 2000
    D. 5000
    Answer: B
    
    The first person to write an essay on aesthetics in China was ____.
    A
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but so is the future of digital racism, as figures from academia, government and the media assert a role for AI in the fight against hate speech and the prevention of violence against Muslims and other communities of colour. But what does AI actually do? Is it being used to create virtual realities that are not inclusive? Or does it have a real potential to accelerate the fight against hate and discrimination? The AI and digital race, 5 questions for the future of AI, asks three researchers who are investigating the relationship between technology and racism, and the intersection between technology and AI.
    The AI and digital racism, 5 questions for the future


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic, or neutral description of your personality]. I enjoy [insert a short, positive, enthusiastic, or neutral description of your hobbies or interests]. I'm always looking for new experiences and opportunities to learn and grow. What's your favorite hobby or activity? I love [insert a short, positive, enthusiastic, or neutral description of your favorite hobby or activity]. I'm always looking for new ways to challenge myself
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and historical center with a rich history dating back to ancient times and a modern city that is known for its fashion, cuisine, and art. It is a major transportation hub and a major tourist destination, with many attractions and events throughout the year. Paris is a city that is a melting pot of cultures and influences, and it is a symbol of France's rich history and culture. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased integration of AI into everyday life: As AI becomes more integrated into our daily lives, we may see more widespread adoption of AI-powered technologies such as voice assistants, self-driving cars, and smart homes.
    
    2. Greater emphasis on ethical and responsible AI: As AI systems become more sophisticated, there will be a growing
    


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
    Generated text:  [Name], and I'm a [describe your profession here]. I'm currently [work or study abroad]. I'm looking to [excuse yourself or what you're interested in doing for a while]. I'm eager to [describe your dream job, hobby, or passion]. What is your name and what are you interested in doing? Let me know when you're ready to connect. [Name] [Phone number] [Email address] [LinkedIn URL] [Twitter username] [Facebook profile link] [Instagram profile link] [Pinterest profile link] [Your website URL]
    I am an AI language model and have not had the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. The city is also known for its vibrant nightlife, stunning architecture, and rich history, making it a popular destination for tourists and locals alike. Paris is a city of contrasts and enchantment, making it a must-visit destination for anyone looking to explore the cultural and historical landmarks of France. (Note: The statement can be further elaborated on by providing more details about the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, as well as noting that Paris is also a cosmopolitan
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of key trends, including:
    
    1. Increased proficiency in natural language processing: As AI becomes more capable of understanding and generating human-like speech and text, it will become increasingly sophisticated in natural language processing.
    
    2. Development of superhuman intelligence: As researchers continue to push the limits of AI, they may develop models that surpass human cognitive abilities, making machines capable of thinking and learning like humans.
    
    3. AI integration with human workers: As AI becomes more integrated into everyday life, it may lead to a more collaborative and automated workplace, where AI systems can perform repetitive tasks and help humans focus on more creative and complex work


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

    'm

     a

     [

    job

     title

    ]

     with

     [

    number

    ]

     years

     of

     experience

     in

     the

     field

     of

     [

    profession

    ].

     I

    've

     always

     been

     fascinated

     by

     [

    idea

    ],

     and

     have

     been

     working

     on

     it

     for

     [

    number

    ]

     years

    .

     I

     have

     a

     passion

     for

     [

    idea

    ],

     and

     I

     strive

     to

     always

     push

     the

     boundaries

     of

     what

    's

     possible

     in

     this

     field

    .

     I

    'm

     excited

     to

     share

     my

     experiences

     and

     learn

     new

     things

    .

     So

    ,

     what

    's

     your

     name

    ?

     How

     can

     I

     get

     to

     know

     you

     better

    ?

     Based

     on

     the

     given

     context

    ,

     what

     other

     details

     or

     information

     would

     be

     relevant

     for

     a

     self

    -int

    roduction

    ?

     The

     given

     context

     states

     that

     the

     character

    's

     name

     is

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Se

    ine

     river

    ,

     with

     a

     population

     of

     around

     

    2

    .

    2

     million

     people

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

     vibrant

     culture

    ,

     and

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     famous

     for

     its

     fashion

     and

     food

     scenes

    ,

     with

     the

     iconic

     fashion

     runway

     being

     a

     symbol

     of

     the

     city

    .

     The

     city

     has

     a

     rich

     cultural

     heritage

     and

     plays

     an

     important

     role

     in

     the

     country

    's

     economy

     and

     identity

    .

     It

     is

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

     and

     is

     a

     popular

     tourist

     destination

    .

     Paris

     is

     a

     major

     cultural

     and

     political

     center

     of

     France

     and

     is

     considered

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     diverse

     and

     exciting

    .

     Here

     are

     some

     of

     the

     possible

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Automation

    :

     With

     the

     continued

     advancement

     of

     AI

    ,

     automation

     is

     expected

     to

     become

     more

     prevalent

     in

     various

     industries

    .

     AI

     systems

     will

     be

     used

     to

     perform

     repetitive

     and

     mundane

     tasks

    ,

     freeing

     up

     human

     workers

     to

     handle

     more

     complex

     and

     creative

     work

    .
    


    2

    .

     Aug

    mented

     Intelligence

    :

     AI

     is

     expected

     to

     become

     even

     more

     integrated

     with

     our

     daily

     lives

    .

     Aug

    mented

     intelligence

     will

     allow

     machines

     to

     interact

     with

     humans

     in

     new

     and

     innovative

     ways

    ,

     such

     as

     through

     speech

     or

     gesture

     recognition

    .
    


    3

    .

     Eth

    ical

     AI

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     there

     will

     be

     increased

     scrutiny

     and

     debate

     about

    



```python
llm.shutdown()
```
