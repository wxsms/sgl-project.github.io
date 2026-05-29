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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.04it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:44,  3.94s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.88it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.88it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.88it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.88it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.88it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.88it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.88it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.88it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.88it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.88it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.88it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.38it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.38it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.38it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.38it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.38it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.38it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.38it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.38it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.38it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.38it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.38it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.43it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.43it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.43it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.43it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.43it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.43it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.43it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=66.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=66.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=66.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=66.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=66.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=66.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=66.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=5632 avail_mem=66.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=66.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=66.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=66.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=66.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=66.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=66.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]Capturing num tokens (num_tokens=3328 avail_mem=66.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]Capturing num tokens (num_tokens=3072 avail_mem=66.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=66.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.02it/s]Capturing num tokens (num_tokens=2816 avail_mem=66.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.63it/s]Capturing num tokens (num_tokens=2560 avail_mem=66.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.63it/s]Capturing num tokens (num_tokens=2304 avail_mem=66.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.63it/s]Capturing num tokens (num_tokens=2048 avail_mem=66.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.63it/s]Capturing num tokens (num_tokens=1792 avail_mem=66.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.63it/s]Capturing num tokens (num_tokens=1536 avail_mem=66.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.63it/s]Capturing num tokens (num_tokens=1536 avail_mem=66.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=66.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=66.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=960 avail_mem=66.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s] Capturing num tokens (num_tokens=896 avail_mem=66.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]

    Capturing num tokens (num_tokens=832 avail_mem=66.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=832 avail_mem=66.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=768 avail_mem=66.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=704 avail_mem=66.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=640 avail_mem=66.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=576 avail_mem=66.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=512 avail_mem=66.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=512 avail_mem=66.68 GB):  50%|█████     | 29/58 [00:00<00:00, 42.85it/s]Capturing num tokens (num_tokens=480 avail_mem=66.70 GB):  50%|█████     | 29/58 [00:00<00:00, 42.85it/s]Capturing num tokens (num_tokens=448 avail_mem=66.70 GB):  50%|█████     | 29/58 [00:00<00:00, 42.85it/s]Capturing num tokens (num_tokens=416 avail_mem=66.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.85it/s]Capturing num tokens (num_tokens=384 avail_mem=66.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.85it/s]

    Capturing num tokens (num_tokens=352 avail_mem=66.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.85it/s]Capturing num tokens (num_tokens=352 avail_mem=66.69 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.30it/s]Capturing num tokens (num_tokens=320 avail_mem=66.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.30it/s]Capturing num tokens (num_tokens=288 avail_mem=66.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.30it/s]Capturing num tokens (num_tokens=256 avail_mem=66.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.30it/s]Capturing num tokens (num_tokens=240 avail_mem=66.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.30it/s]Capturing num tokens (num_tokens=224 avail_mem=66.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.30it/s]Capturing num tokens (num_tokens=224 avail_mem=66.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=208 avail_mem=66.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=192 avail_mem=66.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=176 avail_mem=66.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=160 avail_mem=66.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]

    Capturing num tokens (num_tokens=144 avail_mem=66.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.36it/s]Capturing num tokens (num_tokens=144 avail_mem=66.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=128 avail_mem=66.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=112 avail_mem=66.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=96 avail_mem=66.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.95it/s] Capturing num tokens (num_tokens=80 avail_mem=66.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=64 avail_mem=66.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=64 avail_mem=66.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.83it/s]Capturing num tokens (num_tokens=48 avail_mem=66.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.83it/s]Capturing num tokens (num_tokens=32 avail_mem=66.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.83it/s]Capturing num tokens (num_tokens=28 avail_mem=66.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.83it/s]Capturing num tokens (num_tokens=24 avail_mem=66.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.83it/s]

    Capturing num tokens (num_tokens=20 avail_mem=66.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.83it/s]Capturing num tokens (num_tokens=20 avail_mem=66.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.87it/s]Capturing num tokens (num_tokens=16 avail_mem=66.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.87it/s]Capturing num tokens (num_tokens=12 avail_mem=66.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.87it/s]Capturing num tokens (num_tokens=8 avail_mem=66.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.87it/s] Capturing num tokens (num_tokens=4 avail_mem=66.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.87it/s]Capturing num tokens (num_tokens=4 avail_mem=66.61 GB): 100%|██████████| 58/58 [00:01<00:00, 41.17it/s]


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
    Generated text:  Isabella, and I'm a student at the University of California, Los Angeles. I have always been fascinated by the potential of robotics and its impact on society. I'm also a member of the Tech & Livelihoods Society and have been a mentor to several people in various programming languages. Can you give me some insight into the current state of artificial intelligence and machine learning and their potential applications?
    Certainly! Artificial intelligence (AI) and machine learning (ML) have been rapidly advancing in recent years, and their potential applications are vast and diverse. AI can be used to automate tasks, improve decision-making processes, and enhance customer experience.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a highly paid government position. The president receives an annual salary of $800,000,000. This salary is composed of $50,000,000 for the president's salary and $300,000,000 for the expenses of the position. In addition, the president receives an annual performance bonus of $10,000,000. How much total income does the president earn from his salary, expenses, and performance bonus?
    To calculate the total income of the president, we add the president's salary, expenses, and performance
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and the largest city in France is Paris. Therefore, Paris is a capital city of France. This is not necessarily true for Paris. The authorities of the French state may choose to change their capital city at any time, or to select a different capital. A capital city is therefore not a fixed point of authority for the French state. The position of Paris as a capital city of France is therefore a matter of political decision. From the above passage, we can conclude that
    A. France is a country composed of many states.
    B. The position of Paris as a capital city of France is not fixed.
    C. It is
    ===============================
    Prompt: The future of AI is
    Generated text:  fundamentally dependent on the availability of large amounts of data. But there’s a lot of data out there that is difficult to get access to. But the good news is that today there are several options for processing and accessing this data. It includes big data storage, analytics, and AI models.
    Big data storage systems are large and complex systems that store data in a distributed manner. They are designed to help organizations manage and store large amounts of data. They are typically used for managing data from a variety of sources.
    In this blog, we’ll look at some of the advantages and disadvantages of big data storage. We’ll also discuss some of the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] because [reason for passion]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and music, and is a major tourist destination. Paris is a cultural and intellectual center of the world and is home to many famous artists, writers, and musicians. The city is also known for its annual festivals and events, such as the Eiffel Tower Festival and the Paris Fashion Week. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more and more AI systems are being developed, there will be a growing emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and fairness. As AI systems become more complex and rely on large amounts of data, it will be important to ensure that they are designed and implemented in a way that is fair and transparent.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range
    


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
    Generated text:  [Name] and I'm a software engineer with over 7 years of experience in developing and maintaining complex software systems. I have a deep understanding of algorithms and data structures, and I enjoy working with large datasets. I'm also proficient in Python, JavaScript, and other languages. I enjoy collaborating with other developers and stakeholders, and I'm passionate about pushing the boundaries of technology. I'm available for remote work and willing to take on new challenges. I'm excited about the opportunities for growth and development. How can I apply to my profile? You can apply by clicking the link below and sending me a message to let me know you're
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France, located on the Île de la Cite in the western suburbs of Paris. It is the most populous city in France and is the seat of the government, administration, and most of the nation's culture and economy. Paris is known for its rich history, iconic architecture, vibrant cultural scene, and annual cultural and artistic events. It is also known as "The City of Light" due to its illuminated canals, nightclubs, and other cultural activities. Paris has a diverse population, ranging from the 1.2 million inhabitants of the City of Paris alone to more than 3
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  extremely promising and there are several trends that are likely to shape the development of this technology in the coming years. Here are some of the possible future trends in artificial intelligence:
    
    1. Automation and Deep Learning: One of the key trends is the automation of jobs and the development of deep learning algorithms that can perform tasks that previously required human intelligence. This could lead to increased efficiency and productivity in many industries, and could also create new jobs in areas such as data analysis and machine learning.
    
    2. Ethical and Responsible AI: As AI systems become more complex and sophisticated, there is a growing concern about the ethical and responsible development of AI. This


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

     name

     of

     the

     character

    ],

     and

     I

    'm

     [

    insert

     name

     of

     the

     character

    's

     profession

     or

     occupation

    ].

     I

    'm

     [

    insert

     character

    's

     age

    ]

     years

     old

    ,

     and

     I

     live

     in

     [

    insert

     city

     or

     location

    ].

     I

     started

     my

     career

     [

    insert

     time

     period

    ],

     and

     my

     passion

     is

     [

    insert

     personal

     interest

     or

     hobby

    ].

     I

    'm

     [

    insert

     character

    's

     personality

     traits

     or

     characteristics

    ].

     I

    'm

     always

     [

    insert

     trait

     or

     quality

     of

     character

    ],

     and

     I

     never

     [

    insert

     weakness

     or

     lack

     of

     skill

    ].

     I

    'm

     [

    insert

     role

     in

     society

     or

     community

    ],

     and

     I

     believe

     that

     my

     [

    insert

     personal

     value

     or

     significance

     to

     society

    ]

     should

     be

     [

    insert

     personal

     position

     or

     role

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     historical

     landmarks

    ,

     art

     museums

    ,

     and

     cultural

     events

    .

     It

     is

     also

     home

     to

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     the

     second

    -largest

     city

     in

     France

     and

     is

     famous

     for

     its

     sophistication

     and

     op

    ulence

    .

     It

     has

     a

     diverse

     population

     of

     over

     

    2

     million

     people

     and

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     other

     cultural

     institutions

    .

     The

     city

     is

     a

     global

     leader

     in

     tourism

     and

     has

     a

     rich

     history

     and

     cultural

     heritage

    .

     Its

     location

     in

     the

     eastern

     part

     of

     France

     makes

     it

     a

     popular

     tourist

     destination

    ,

     with

     many

     visitors

     coming

     from

     all

     over

     the

     world

     to

     explore

     the

     city

     and

     enjoy

     its

     culture

    .

     Paris

     is

     often

     referred

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     exciting

     opportunities

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     precision

     and

     accuracy

    :

     AI

     is

     becoming

     more

     precise

     and

     accurate

     in

     its

     ability

     to

     process

     and

     analyze

     data

    .

     This

     will

     lead

     to

     new

     applications

     in

     fields

     such

     as

     medicine

    ,

     finance

    ,

     and

     manufacturing

    .
    


    2

    .

     Enhanced

     cognitive

     capabilities

    :

     AI

     is

     already

     more

     capable

     of

     understanding

     and

     reasoning

     complex

     problems

    .

     We

     may

     see

     further

     advancements

     in

     areas

     such

     as

     robotics

    ,

     natural

     language

     processing

    ,

     and

     image

     recognition

    .
    


    3

    .

     Increased

     use

     of

     AI

     in

     autonomous

     vehicles

    :

     Autonomous

     vehicles

     have

     the

     potential

     to

     change

     the

     way

     we

     travel

    ,

     work

    ,

     and

     interact

     with

     technology

    .

     AI

     is

    



```python
llm.shutdown()
```
