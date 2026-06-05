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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.25it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.25it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.39it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.09 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.06 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.06 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=59.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=960 avail_mem=59.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s] Capturing num tokens (num_tokens=896 avail_mem=59.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]

    Capturing num tokens (num_tokens=832 avail_mem=59.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=832 avail_mem=59.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=768 avail_mem=59.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=704 avail_mem=59.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=640 avail_mem=59.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=576 avail_mem=59.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=512 avail_mem=58.99 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=512 avail_mem=58.99 GB):  50%|█████     | 29/58 [00:00<00:00, 43.45it/s]Capturing num tokens (num_tokens=480 avail_mem=59.00 GB):  50%|█████     | 29/58 [00:00<00:00, 43.45it/s]Capturing num tokens (num_tokens=448 avail_mem=59.00 GB):  50%|█████     | 29/58 [00:00<00:00, 43.45it/s]Capturing num tokens (num_tokens=416 avail_mem=58.97 GB):  50%|█████     | 29/58 [00:00<00:00, 43.45it/s]

    Capturing num tokens (num_tokens=384 avail_mem=58.96 GB):  50%|█████     | 29/58 [00:00<00:00, 43.45it/s]Capturing num tokens (num_tokens=352 avail_mem=58.96 GB):  50%|█████     | 29/58 [00:00<00:00, 43.45it/s]Capturing num tokens (num_tokens=352 avail_mem=58.96 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.80it/s]Capturing num tokens (num_tokens=320 avail_mem=58.95 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.80it/s]Capturing num tokens (num_tokens=288 avail_mem=58.95 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.80it/s]Capturing num tokens (num_tokens=256 avail_mem=58.95 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=240 avail_mem=58.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.80it/s]

    Capturing num tokens (num_tokens=224 avail_mem=58.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.80it/s]Capturing num tokens (num_tokens=224 avail_mem=58.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=208 avail_mem=58.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=192 avail_mem=58.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=176 avail_mem=58.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=160 avail_mem=58.93 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=144 avail_mem=58.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=144 avail_mem=58.92 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=128 avail_mem=58.92 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=112 avail_mem=58.92 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=96 avail_mem=58.92 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.63it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=58.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=64 avail_mem=58.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=64 avail_mem=58.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.19it/s]Capturing num tokens (num_tokens=48 avail_mem=58.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.19it/s]Capturing num tokens (num_tokens=32 avail_mem=58.90 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.19it/s]Capturing num tokens (num_tokens=28 avail_mem=58.90 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.19it/s]Capturing num tokens (num_tokens=24 avail_mem=58.90 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.19it/s]Capturing num tokens (num_tokens=20 avail_mem=58.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.19it/s]Capturing num tokens (num_tokens=20 avail_mem=58.89 GB):  93%|█████████▎| 54/58 [00:01<00:00, 37.49it/s]Capturing num tokens (num_tokens=16 avail_mem=58.89 GB):  93%|█████████▎| 54/58 [00:01<00:00, 37.49it/s]Capturing num tokens (num_tokens=12 avail_mem=58.89 GB):  93%|█████████▎| 54/58 [00:01<00:00, 37.49it/s]

    Capturing num tokens (num_tokens=8 avail_mem=58.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 37.49it/s] Capturing num tokens (num_tokens=4 avail_mem=58.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 37.49it/s]Capturing num tokens (num_tokens=4 avail_mem=58.88 GB): 100%|██████████| 58/58 [00:01<00:00, 36.67it/s]


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
    Generated text:  Tanya and I am a member of the team of scientists who study the effects of sunshine on the living things in the forest. I am also one of the researchers on the research group that uses artificial intelligence to find out how much sunshine is needed in a forest to support the life of the forest.
    Trees can survive without sunlight. They can be quite cold and wet, but they can survive by absorbing sunlight. The tree absorbs sunlight in two ways: photo-synthesis and transpiration.
    If you look at an image of a tree, you can see that a tree has chloroplasts in its cells. These chloroplasts are responsible for
    ===============================
    Prompt: The president of the United States is
    Generated text:  now considering whether to go on the foreign mission that the Senate is considering. She has been told by his national security adviser that there are two possible options. 
    
    Option A: The president can choose to send an ambassador to another nation, or she can decide to continue with the mission without a new ambassador. 
    
    Option B: The president can choose to increase the budget for the mission, or she can decide not to continue with the mission.
    
    The president has been asked to make a decision between these two options. She has no other information about the outcome of the mission. She has two options to consider. 
    
    What type of decision making strategy
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. Madrid
    C. Rome
    D. Athens
    答案：A
    解析：巴黎是法国的首都，位于法国东南部的塞纳河畔，是法国的首府。巴黎是法国的首都，位于法国东南部的塞纳河畔，是法国的首府。巴黎是法国的首都，位于法国东南部的塞纳河畔，是法国的首府。巴黎是法国的首都，位于法国东南部的塞纳河畔，是法国的首府。巴黎是法国的首都，位于法国东南部的塞纳河畔
    ===============================
    Prompt: The future of AI is
    Generated text:  about making the world a more beautiful place. In the end, AI will bring us more beauty.
    A. beautiful
    B. beautiful
    C. beauty
    D. beautiful
    Answer:
    
    B
    
    Please select the word from the following options that is of the same type as the given word:
    A. hand
    B. foot
    C. foot's
    D. fingers
    Answer:
    
    D
    
    The concept of innovation is seen as a noble pursuit that should be pursued in every field of society. In the development of the real economy, there is a trend of innovation being heavily promoted. This is mainly due to the rapid growth of new


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working in this field for [number of years] years. I am passionate about [reason for interest in the field]. I am always looking for ways to [action or goal]. I am [age] years old. I am [gender] and I am [race or ethnicity]. I am [occupation] and I am [language]. I am [religion or belief]. I am [country of origin]. I am [country of citizenship]. I am [country of residence]. I am [country of origin]. I am [country of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament, the French National Museum, and the Louvre Museum. Paris is a bustling city with a rich history and culture, and it is a popular tourist destination. The city is known for its fashion, art, and cuisine, and it is a major center for business and finance in Europe. Paris is a city that is constantly evolving and changing, with new developments and attractions being added regularly. The city is also home to many international organizations and institutions, including the European
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, and it has the potential to revolutionize the field by improving diagnostic accuracy and personalized treatment plans.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes, reduce waste, and improve quality control. As AI technology continues to advance, we can expect to see even more widespread adoption in manufacturing.
    
    3. AI in transportation: AI is already being used in transportation to
    


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
    Generated text:  [Your Name], and I'm a [career] specializing in [specific field]. What can you tell me about yourself? [Your Name], a [career] specializing in [specific field], is a passionate and experienced professional who is dedicated to [job title]. I have over [number of years of experience] years of experience in [specific area], and I am always eager to learn and grow my skills. Additionally, I am a person who is [positive, enthusiastic, hardworking, etc.] and I am always looking for ways to improve my abilities and contribute to my team. What would you like me to do next? [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The statement is: Paris is the capital city of France. 
    
    This concise statement encapsulates the essential fact that Paris is the main city of France, providing a clear and factual overview of its role and importance in the nation. However, a more elaborate response might include the other facts about Paris, such as its historical significance, cultural influence, and role in France's modern society. 
    
    For example, it could be stated: "Paris, founded by the French crown in 973, has been the center of French culture, politics, and diplomacy for over a thousand years. It is the capital of France, known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by increased automation, integration with human expertise, and ethical considerations. Some possible future trends in AI include:
    
    1. Increased automation: AI will continue to automate repetitive and mundane tasks, but it will also be used to perform creative, strategic, and creative tasks. For example, AI-powered chatbots will be used to provide customer service, but they will also be used to help optimize supply chain processes.
    
    2. Increased integration with human expertise: AI will continue to integrate with human expertise, allowing for more sophisticated problem-solving and decision-making. For example, AI-powered systems will be able to analyze and interpret large amounts of data


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

    position

    ]

     for

     [

    company

     name

    ].

     I

     have

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     [

    industry

    ].

     In

     my

     previous

     role

    ,

     I

     had

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     [

    industry

    ].

     I

     graduated

     [

    highest

     degree

    ]

     from

     [

    un

    iversity

     or

     school

    ].

     I

     have

     a

     passion

     for

     [

    interest

     or

     hobby

    ]

     and

     I

     enjoy

     [

    something

     about

     myself

     that

     sets

     me

     apart

    ].

     I

     am

     [

    a

     characteristic

     or

     trait

     that

     defines

     me

     as

     a

     character

    ],

     and

     I

     strive

     to

     [

    what

     I

     believe

     is

     my

     goal

     or

     purpose

     as

     a

     character

    ].

     I

     am

     excited

     to

     [

    time

     period

     or

     project

    ]

     and

     look

     forward

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     located

     in

     the

     south

     of

     the

     country

     and

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     rich

     cultural

     heritage

     and

     romantic

     ambiance

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     one

     of

     the

     world

    's

     most

     important

     cultural

    ,

     economic

    ,

     and

     political

     centers

    .

     Located

     on

     the

     left

     bank

     of

     the

     Se

    ine

     River

    ,

     Paris

     is

     the

     political

     and

     economic

     capital

     of

     France

    ,

     and

     the

     home

     of

     the

     French

     parliament

    .

     It

     is

     also

     home

     to

     many

     famous

     landmarks

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

     the

     Palace

     of

     Vers

    ailles

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     a

     city

     that

     is

     known

     for

     its

     gastr

    onomy

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

    ,

     and

     it

    's

     likely

     to

     see

     significant

     changes

     in

     the

     coming

     years

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

     collaboration

     between

     humans

     and

     AI

    :

     As

     AI

     continues

     to

     improve

    ,

     it

    's

     likely

     to

     become

     more

     integrated

     with

     humans

    ,

     allowing

     for

     greater

     collaboration

     and

     innovation

    .
    


    2

    .

     Improved

     privacy

     and

     security

    :

     AI

     is

     becoming

     more

     sophisticated

    ,

     and

     there

    's

     a

     growing

     concern

     about

     the

     security

     and

     privacy

     of

     the

     data

     it

     processes

    .

     There

     will

     be

     more

     effort

     put

     into

     protecting

     the

     privacy

     and

     security

     of

     AI

     systems

    .
    


    3

    .

     Increased

     automation

    :

     AI

     is

     likely

     to

     become

     more

     prevalent

     in

     many

     industries

    ,

     from

     manufacturing

     to

     transportation

     to

     healthcare

    .

     This

     automation

     will

    



```python
llm.shutdown()
```
