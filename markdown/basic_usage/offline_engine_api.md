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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.16it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:12,  3.71it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:12,  3.71it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.71it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.71it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:12,  3.71it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.31it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.31it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.31it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.31it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.31it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.31it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.31it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.31it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.31it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:05<00:02, 13.37it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 20.21it/s]

    Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 28.09it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 28.09it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 28.09it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 28.09it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 28.09it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 28.09it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 28.09it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 28.09it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 28.09it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 28.09it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 28.09it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 37.89it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 37.89it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 37.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   9%|▊         | 5/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.12 GB):   9%|▊         | 5/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.36it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.59it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 28.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 28.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.08 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.21it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.63 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.33it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.33it/s] Capturing num tokens (num_tokens=896 avail_mem=74.64 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.33it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.33it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.33it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.33it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.78it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.78it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.78it/s]Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.78it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.78it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.78it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=384 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.89it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.78it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  71%|███████   | 41/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=112 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 42.91it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.58 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.16it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.92it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.92it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.92it/s]Capturing num tokens (num_tokens=20 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.92it/s]Capturing num tokens (num_tokens=16 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.92it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.92it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.04it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.04it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.04it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 38.20it/s]


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
    Generated text:  Luke. I have been reading this story and it's been quite an experience. After having read it, I feel like it's something very special. I want to write a short summary of the story, but I'm not sure how to begin. Can you provide a brief summary of the story? 
    
    First, I have a list of questions that I have about the story. These questions are as follows:
    
    1. What is the main character's name?
    2. What is the setting?
    3. What is the main conflict?
    4. What is the climax?
    5. What is the resolution?
    
    Once I have the answers to these
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office that can be held by either a man or a woman. In 2004, Hillary Clinton was the first woman to win a presidential election. The president of the United States was re-elected in 2008, 2012, and 2016, and the last time a woman won the presidency was in 1992, when Nancy Reagan was re-elected. 
    
    If the president of the United States was elected in 1992, and Hillary Clinton became the first woman to win the presidency, what is the probability that the next election will result in a
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Versailles
    C. Oxford
    D. Dublin
    Answer:
    
    A
    
    Which of the following is NOT a common cause of acute cholecystitis? A. Gallstones B. Gallbladder tumor C. Gallstones + gallbladder tumor D. Gallstones + gallbladder tumor + infection
    Answer:
    
    C
    
    Which of the following is NOT an example of an ecosystem? 
    A. A forest
    B. A river
    C. A country
    D. A school
    Answer:
    
    C
    
    Which of the following is not a component of an ecosystem?
    A. Sunlight
    
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be very different, and for the same reason that AI is going to be different today. We are going to see a huge shift in the future of AI, and this is going to be a positive shift, as it will allow us to do things that we can't do today.
    AI in the future will be an even more important part of our lives. But how can we prepare for this future when the future is uncertain and we do not know what it will be? One way is to research the latest AI trends and technologies, and to keep abreast of the latest developments. You can also stay informed about AI news and


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has always been [Attraction/Interest/Challenge] to me. I'm always looking for [What I'm Looking For] and I'm always eager to learn new things. I'm a [What I'm Proud of] and I'm always [What I'm Proud of]. I'm a [What I'm Proud of] and I'm always [What I'm Proud of]. I'm a [What I'm Proud of] and I'm always [What I'm Proud of]. I'm a [What I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its rich history, beautiful architecture, and vibrant culture. It is also the world's largest city, with a population of over 2.5 million people. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a popular tourist destination, with millions of visitors each year. Paris is a city that is constantly evolving, with new developments and cultural events taking place throughout the year. The city is also known for its cuisine, with many famous French dishes such as croissants, boudin, and escargot.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI systems, there is a growing emphasis on ethical AI. This includes developing AI systems that are transparent, accountable, and responsible for their actions. The development of ethical AI will likely involve a greater focus on ensuring that AI systems are designed and implemented in a way that is fair, unbiased, and transparent.
    
    2. Increased use of AI
    


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
    Generated text:  [Your Name], and I'm an AI language model. As an AI, I'm here to help you with any questions you have, answer any queries you may have, and provide information to the best of my abilities. I can also answer any questions you may have, and I'm here to assist you whenever you need it. So, if you have any questions or need help, don't hesitate to reach out to me. I'm always here to assist you! 📚✨ #ChatGPT #AI #LanguageModeler #AIExpert #ChatGPTFan #IntroducingMyself #SkillfulAI #G
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, often referred to as the “City of a Thousand Castles.” It is a popular tourist destination known for its beautiful architecture, rich history, and vibrant cultural scene. The city is also home to the Eiffel Tower, the Louvre Museum, and other notable landmarks. Paris has a rich cultural heritage and is a major economic center in Europe. The city is also known for its unique cuisine and fashion industry. As of 2021, Paris had a population of approximately 2.1 million people. The city is often described as a city of contrasts, with its historical landmarks and modern architecture blending together to create
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but some trends are likely to shape it. Here are a few:
    
    1. Increased use of AI in healthcare: AI is already being used in medical diagnosis and treatment, and it has the potential to revolutionize the field. AI can help doctors identify early signs of disease, predict patient outcomes, and optimize treatment plans. However, it also raises ethical concerns about privacy, data security, and bias in AI algorithms.
    
    2. Advancements in AI ethics and regulation: AI is a complex technology that can have both positive and negative effects on society. There is a growing need for ethical guidelines and regulations to govern AI development, deployment,


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

     am

     a

     self

    -pro

    claimed

     AI

     language

     model

    .

     How

     can

     I

     assist

     you

     today

    ?

     Let

     me

     know

     if

     you

     have

     any

     questions

     or

     need

     any

     information

    .

     I

     am

     always

     here

     to

     help

    !

     [

    Name

    ]

     (

    AI

     language

     model

    )

     Hey

    !

     [

    Name

    ],

     I

    'm

     [

    Name

    ].

     How

     can

     I

     assist

     you

     today

    ?

     [

    Name

    ]

     (

    AI

     language

     model

    )

     Hey

    !

     [

    Name

    ],

     I

    'm

     [

    Name

    ].

     How

     can

     I

     assist

     you

     today

    ?

     [

    Name

    ]

     (

    AI

     language

     model

    )

     Hey

    !

     [

    Name

    ],

     I

    'm

     [

    Name

    ].

     How

     can

     I

     assist

     you

     today

    ?

     [

    Name

    ]

     (

    AI

     language

     model

    )

     Hey

    !

     [

    Name

    ],

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

    ,

     near

     the

     E

    iff

    el

     Tower

    ,

     and

     is

     known

     for

     its

     rich

     history

    ,

     cultural

     institutions

    ,

     and

     famous

     landmarks

     such

     as

     the

     Lou

    vre

     Museum

    .

     It

     is

     also

     home

     to

     many

     international

     institutions

     and

     is

     a

     cultural

     center

     for

     France

    .

     Paris

     is

     the

     

    1

    1

    th

     largest

     city

     in

     the

     world

     by

     population

     and

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     a

     Hundred

     Faces

    ."

     The

     city

     is

     known

     for

     its

     fine

     dining

    ,

     art

    ,

     and

     fashion

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     a

     diverse

     population

     and

     is

     considered

     one

     of

     the

     most

     exciting

     cities

     in

     the

     world

    .

     The

     city

     is

     also

     known

     for

     its

     lively

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     promising

     and

     has

     the

     potential

     to

     revolution

    ize

     many

     industries

    ,

     improve

     our

     quality

     of

     life

    ,

     and

     solve

     complex

     problems

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

     Enhanced

     Real

    -Time

     Intelligence

    :

     AI

     is

     becoming

     more

     accurate

     and

     real

    -time

    .

     As

     AI

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     more

     intelligent

     machines

     interacting

     with

     us

     in

     real

    -time

    ,

     in

     everything

     from

     cars

     to

     healthcare

     to

     education

    .
    


    2

    .

     Autonomous

     Vehicles

    :

     As

     autonomous

     vehicles

     become

     more

     advanced

    ,

     we

     can

     expect

     to

     see

     them

     operating

     on

     our

     roads

     and

     in

     our

     homes

    .

     This

     will

     reduce

     the

     number

     of

     accidents

     and

     improve

     safety

    .
    


    3

    .

     Bi

    ometric

     Recognition

    :

     Bi

    ometric

     recognition

     is

     becoming

    



```python
llm.shutdown()
```
