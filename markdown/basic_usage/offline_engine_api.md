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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.16it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.27it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.45it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.41it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.41it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.41it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.32it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.32it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.32it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.32it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.32it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.32it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.32it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.08it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.61it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.61it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.61it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.05it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.05it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.01 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.01 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.01 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.00 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.00 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.69 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=62.01 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.64it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=62.01 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=62.01 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=62.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.09it/s]Capturing num tokens (num_tokens=1280 avail_mem=62.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.99 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.09it/s]Capturing num tokens (num_tokens=960 avail_mem=62.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.09it/s] Capturing num tokens (num_tokens=896 avail_mem=62.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.09it/s]Capturing num tokens (num_tokens=832 avail_mem=62.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.09it/s]Capturing num tokens (num_tokens=832 avail_mem=62.00 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.46it/s]Capturing num tokens (num_tokens=768 avail_mem=61.99 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.46it/s]

    Capturing num tokens (num_tokens=704 avail_mem=61.99 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.46it/s]Capturing num tokens (num_tokens=640 avail_mem=61.99 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.46it/s]Capturing num tokens (num_tokens=576 avail_mem=61.99 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.46it/s]Capturing num tokens (num_tokens=512 avail_mem=61.97 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.46it/s]Capturing num tokens (num_tokens=512 avail_mem=61.97 GB):  50%|█████     | 29/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=480 avail_mem=61.99 GB):  50%|█████     | 29/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=448 avail_mem=61.98 GB):  50%|█████     | 29/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=416 avail_mem=61.98 GB):  50%|█████     | 29/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=384 avail_mem=61.98 GB):  50%|█████     | 29/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=352 avail_mem=61.97 GB):  50%|█████     | 29/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=352 avail_mem=61.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=320 avail_mem=61.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.72it/s]

    Capturing num tokens (num_tokens=288 avail_mem=61.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=256 avail_mem=61.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=240 avail_mem=61.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=224 avail_mem=61.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=224 avail_mem=61.96 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.14it/s]Capturing num tokens (num_tokens=208 avail_mem=61.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.14it/s]Capturing num tokens (num_tokens=192 avail_mem=61.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.14it/s]Capturing num tokens (num_tokens=176 avail_mem=61.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.14it/s]Capturing num tokens (num_tokens=160 avail_mem=61.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.14it/s]

    Capturing num tokens (num_tokens=144 avail_mem=61.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.14it/s]Capturing num tokens (num_tokens=144 avail_mem=61.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=128 avail_mem=61.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=112 avail_mem=61.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=96 avail_mem=61.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.01it/s] Capturing num tokens (num_tokens=80 avail_mem=61.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=64 avail_mem=61.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.01it/s]Capturing num tokens (num_tokens=64 avail_mem=61.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.44it/s]Capturing num tokens (num_tokens=48 avail_mem=61.92 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.44it/s]Capturing num tokens (num_tokens=32 avail_mem=61.92 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.44it/s]Capturing num tokens (num_tokens=28 avail_mem=61.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.44it/s]

    Capturing num tokens (num_tokens=24 avail_mem=61.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.44it/s]Capturing num tokens (num_tokens=20 avail_mem=61.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.44it/s]Capturing num tokens (num_tokens=20 avail_mem=61.91 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.92it/s]Capturing num tokens (num_tokens=16 avail_mem=61.91 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.92it/s]Capturing num tokens (num_tokens=12 avail_mem=61.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.92it/s]Capturing num tokens (num_tokens=8 avail_mem=61.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.92it/s] Capturing num tokens (num_tokens=4 avail_mem=61.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.92it/s]Capturing num tokens (num_tokens=4 avail_mem=61.90 GB): 100%|██████████| 58/58 [00:01<00:00, 35.69it/s]


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
    Generated text:  Ido. My head is like a fish, and my body is like a crab. My head is made of a pair of big eyes, and a pair of big ears. My head is like a fish because it can see in all directions, and it can hear everything in the world. My body is like a crab because it can move and can swim in water. So I am a crab. Please tell me something about you. (Multi-choice question)
    A) I am a fish.  
    B) I am a crab.  
    C) I am a bird.  
    D) I am an animal. The correct answer is
    ===============================
    Prompt: The president of the United States is
    Generated text:  a position that is often subject to scrutiny and public opinion. To ensure the president remains accountable and serves the interests of the American people, the U. S. government has a set of rules that govern how the president can be removed from office. One of the most important rules in this system is the "recall" process, which allows citizens to remove a president from office if they believe the president is not serving in the country's best interests.
    
    The process for a recall is quite simple. The president must be found guilty of violating the law or unethical behavior that would likely lead to the president's removal. The process for a recall in the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris was the ancient capital of France. The construction of the bridge between the city and the island of Corsica began in 1960, but it was not opened officially until 1967. This bridge is made of steel. It has a length of 2.5 kilometers. A man can travel from Paris to the island of Corsica through the bridge in about 15 minutes. The bridge was built to make people travel between the two cities without having to pass by the island of Corsica. The bridge has a number of good places to watch the people on the island of Corsica walking.
    ===============================
    Prompt: The future of AI is
    Generated text:  young and in many ways exciting. The technology is rapidly advancing and is having an increasingly powerful impact on business and society. At the same time, the applications of AI are reaching out further into the future, as a powerful open source project like the Linux Foundation has endorsed the use of open source code and software.
    But as with any technology, there is a lot of buzz about AI technology – and often the media, but also the public, are not always clear about what it is. So the following information will help to put the technology in context – to give you a better understanding of the technology and the impact it is having. The information


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your job or experience here]. What do you do for a living? I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field. What do you enjoy doing in your free time? I enjoy spending time with my family and friends, reading books, and trying new foods. What do you think makes you unique? I think my ability to learn and adapt quickly
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for finance, business, and tourism in Europe. It is a popular tourist destination and a cultural hub for France and the world. The city is home to many important institutions such as the French Academy of Sciences and the French National Library. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ensuring that AI systems are developed and used in a way that is fair, transparent, and accountable.
    
    2. Greater integration with human decision-making: AI is likely to become more integrated with human decision-making processes, allowing for more complex and nuanced decision-making.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As
    


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
    Generated text:  [Your Name], and I am an [Age] year old [Occupation]. I have always been [What you consider to be your strong point] and I enjoy [What you like to do]. I also have a bit of [What you consider to be your weakness], but I always try to [How you overcome this]. I am always looking for [What you like to do to learn new things]. I am looking forward to [What you like to do for work]. I am eager to [What you like to do in your free time]. I am looking forward to [What you want to do for your family]. My
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    I apologize, but there seems to be a misunderstanding. Paris is the capital and largest city of France. It is not a separate city. Paris is the capital of France, not a separate city. The capital of France is Paris. Is there anything else I can help you with?
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly dynamic and multifaceted, with a wide range of possible trends and advancements. Here are some of the key trends that could shape the future of AI:
    
    1. Increased Efficiency and Personalization: AI is expected to become increasingly intelligent and capable of adapting to new situations and environments. This could lead to increased efficiency across a wide range of industries, as AI-powered systems are expected to perform tasks more quickly and accurately than human workers.
    
    2. Enhanced Collaboration and Communication: As AI becomes more integrated into our daily lives, we're likely to see an increase in collaboration and communication between humans and AI. This could lead to more


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

    ’m

     a

     [

    Age

    ]

     year

    -old

     girl

     with

     [

    Occup

    ation

    ]

     who

     is

     passionate

     about

     [

    Inter

    ests

    ].

     I

     love

     spending

     my

     free

     time

     exploring

     the

     world

     and

     meeting

     new

     people

    .

     I

     have

     a

     strong

     sense

     of

     justice

     and

     will

     always

     strive

     to

     help

     others

     and

     make

     the

     world

     a

     better

     place

    .

     I

    ’m

     a

     true

     believer

     in

     the

     power

     of

     positivity

     and

     hope

    ,

     and

     I

     hope

     to

     inspire

     others

     to

     do

     the

     same

    .

     I

    ’m

     always

     eager

     to

     learn

     new

     things

     and

     try

     new

     things

    ,

     and

     I

    ’m

     a

     natural

     leader

     who

     thr

    ives

     on

     a

     good

     team

     and

     a

     supportive

     environment

    .

     I

    ’m

     confident

     in

     my

     abilities

     and

     always

     strive

     to

     achieve

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Its

     population

     is

     over

     one

     million

     and

     it

     is

     the

     largest

     city

     in

     Europe

    .

     It

     is

     known

     for

     its

     art

     and

     culture

    ,

     with

     museums

    ,

     galleries

    ,

     theaters

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     which

     is

     a

     significant

     contributor

     to

     the

     city

    's

     economy

     and

     culture

    .

     The

     city

     is

     also

     home

     to

     the

     French

     Parliament

    ,

     the

     country

    's

     legislative

     body

    ,

     and

     is

     the

     seat

     of

     the

     President

     of

     the

     Republic

    .

     Additionally

    ,

     it

     is

     a

     popular

     tourist

     destination

    ,

     with

     over

     

    1

    0

     million

     visitors

     annually

    .

     Paris

     is

     a

     major

     hub

     of

     the

     French

     economy

     and

     culture

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     dynamic

     and

     unpredictable

    ,

     with

     potential

     to

     bring

     about

     significant

     changes

     in

     various

     fields

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

     AI

     will

     become

     more

     pervasive

    :

     As

     AI

     continues

     to

     advance

    ,

     it

     will

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     from

     smart

     home

     devices

     to

     self

    -driving

     cars

    .

     AI

     will

     also

     become

     more

     accessible

     to

     individuals

    ,

     with

     more

     people

     having

     access

     to

     the

     technology

     and

     tools

     necessary

     to

     use

     it

    .
    


    2

    .

     AI

     will

     be

     more

     versatile

    :

     As

     AI

     continues

     to

     learn

     and

     improve

    ,

     it

     will

     become

     more

     capable

     of

     performing

     a

     wide

     range

     of

     tasks

    .

     This

     means

     that

     AI

     will

     be

     able

     to

     perform

     a

     wider

     range

     of

     tasks

    ,

     from

     data

     analysis

    



```python
llm.shutdown()
```
