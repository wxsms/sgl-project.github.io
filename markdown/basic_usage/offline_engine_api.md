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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.41it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.89it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 23.99it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 23.99it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.99it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.99it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.30 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.30 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.30 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.30 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.30 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.29 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.28 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.28 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.28 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.27 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.27 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.27 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.26 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=53.26 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.26 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.26 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.25 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.25 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.25 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.24 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.24 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.24 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.22 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=960 avail_mem=53.24 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s] Capturing num tokens (num_tokens=896 avail_mem=53.23 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]

    Capturing num tokens (num_tokens=832 avail_mem=53.23 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.97it/s]Capturing num tokens (num_tokens=832 avail_mem=53.23 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=768 avail_mem=53.23 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=704 avail_mem=53.23 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=640 avail_mem=53.22 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=576 avail_mem=53.22 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=512 avail_mem=53.21 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.98it/s]Capturing num tokens (num_tokens=512 avail_mem=53.21 GB):  50%|█████     | 29/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=480 avail_mem=53.22 GB):  50%|█████     | 29/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=448 avail_mem=53.22 GB):  50%|█████     | 29/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=416 avail_mem=53.22 GB):  50%|█████     | 29/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=384 avail_mem=53.22 GB):  50%|█████     | 29/58 [00:00<00:00, 43.36it/s]

    Capturing num tokens (num_tokens=352 avail_mem=53.21 GB):  50%|█████     | 29/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=352 avail_mem=53.21 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=320 avail_mem=53.20 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=288 avail_mem=53.20 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=256 avail_mem=53.20 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=240 avail_mem=53.20 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=224 avail_mem=53.19 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=224 avail_mem=53.19 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.03it/s]Capturing num tokens (num_tokens=208 avail_mem=53.19 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.03it/s]Capturing num tokens (num_tokens=192 avail_mem=53.19 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.03it/s]Capturing num tokens (num_tokens=176 avail_mem=53.18 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.03it/s]Capturing num tokens (num_tokens=160 avail_mem=53.18 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.03it/s]

    Capturing num tokens (num_tokens=144 avail_mem=53.18 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.03it/s]Capturing num tokens (num_tokens=144 avail_mem=53.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=128 avail_mem=53.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=112 avail_mem=53.17 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=96 avail_mem=53.17 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.76it/s] Capturing num tokens (num_tokens=80 avail_mem=53.17 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=64 avail_mem=53.16 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=64 avail_mem=53.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=48 avail_mem=53.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=32 avail_mem=53.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=28 avail_mem=53.15 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=24 avail_mem=53.15 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.16it/s]

    Capturing num tokens (num_tokens=20 avail_mem=53.15 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.16it/s]Capturing num tokens (num_tokens=20 avail_mem=53.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=16 avail_mem=53.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=12 avail_mem=53.14 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=8 avail_mem=53.14 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.76it/s] Capturing num tokens (num_tokens=4 avail_mem=53.13 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=4 avail_mem=53.13 GB): 100%|██████████| 58/58 [00:01<00:00, 41.55it/s]


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
    Generated text:  Matt and I am a teacher at K-12. I teach Literature, English, Social Studies, and Science, with a passion for Teaching and Learning. I have always been a fan of reading and writing. I became a teacher after I fell in love with literature and reading. I believe that the goal of education is to help students to grow into well-rounded, responsible citizens with a love for learning. As a teacher, my goal is to motivate my students to understand the world around them, while teaching them to think critically, and to form their own opinions and perspectives. My passion for teaching is rooted in my love for learning,
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to visit the moon or the island of Java. He knows that the distance from the moon to the island of Java is 30,000 miles. If he visits the moon, he will spend 30 days there. If he visits Java, he will spend 5 days on the island and then 15 days back on the moon. How many days will he spend on the moon if he chooses the island of Java?
    
    To determine how many days the president of the United States will spend on the moon if he chooses the island of Java, we need to follow these steps:
    
    1. Identify
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Beijing
    D. Moscow
    Answer: A
    
    The main factor that determines the size of the urban land use area of an urban area is ____.
    A. The proportion of urban land use in total land area
    B. The proportion of various types of land use in total land area
    C. The area of urban land use
    D. The proportion of various types of urban land use in total urban land area
    Answer: B
    
    For a certain construction project, the total investment amount is 500 million yuan. The bank loan amount is 300 million
    ===============================
    Prompt: The future of AI is
    Generated text:  not at all clear. In the past, it was a bold goal to implement a quantum computer to break the cryptographic system, but in the last decade, it is possible to simulate a quantum computer through the use of quantum computers. The use of quantum computers has the potential to solve problems that cannot be solved through classical computers.
    
    In a video, Martin is explaining the use of quantum computers in the future of AI. He mentions that the use of quantum computers in AI is not just a matter of a new technology, but also requires a different approach. The video goes on to say that the use of quantum computers is a departure from the existing


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


    Generated text:  [Name] and I'm a [occupation] with [number] years of experience in [field]. I'm a [type of person] who is always [positive trait]. I'm [age] years old and I'm [gender]. I'm [occupation] with [number] years of experience in [field]. I'm a [type of person] who is always [positive trait]. I'm [age] years old and I'm [gender]. I'm [occupation] with [number] years of experience in [field]. I'm a [type of person] who is always [positive trait]. I'm [age
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic center with a rich history dating back to the Roman Empire and the French Revolution. It is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its diverse cuisine, fashion, and art scene. Paris is a major hub for international trade and diplomacy, and is a major tourist destination. It is also a major center for French politics and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, but there is a growing trend towards more personalized and accurate diagnoses and treatment plans.
    
    4. Greater use of AI in
    


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
    Generated text:  [Name], and I'm a [job title] at [company name]. I'm currently [number of years] years of experience. My favorite hobby is [any hobby I enjoy]. My first career goal is to [insert short, self-imposed, but achievable, goal]. What's your favorite hobby? I'm a fan of [insert any hobby you like]. What is your favorite hobby? I like [insert any hobby you like]. How did you get into your current career? My mom introduced me to [insert your great-grandparents' profession], and I've been fascinated by the world of [insert a profession they might
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historic and cultural center with many iconic landmarks, including Notre Dame Cathedral and the Louvre Museum. Paris has a diverse population and is known for its innovative architecture and fashion industry. The city is also home to many museums, art galleries, and theaters. Despite its size, Paris has a vibrant and multicultural community that attracts people from all over the world. The city is known for its festivals, such as the La Fête de la Musique (Music Festival), which attracts hundreds of thousands of visitors annually. Overall, Paris is a vibrant and exciting city that is a must-visit destination for anyone interested in history,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, with potential to push the boundaries of what it can do. Here are some possible future trends in AI:
    
    1. Personalized AI: AI is becoming increasingly accurate at predicting human behavior and preferences, allowing for more personalized experiences. This could lead to more efficient and effective decision-making, as well as a greater ability to understand and adapt to individual differences.
    
    2. Autonomous AI: AI is becoming more capable of performing tasks without human intervention, but it is unlikely to fully replace humans in the workplace. However, it could potentially play a more supportive role, providing assistance and guidance where necessary.
    
    3. Cybersecurity AI: AI is


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

    Your

     Name

    ],

     a

     

    3

    2

    -year

    -old

     software

     engineer

     with

     a

     deep

     passion

     for

     technology

     and

     innovation

    .

     With

     over

     ten

     years

     of

     experience

     in

     the

     industry

    ,

     I

     bring

     a

     unique

     set

     of

     skills

     and

     perspectives

     to

     the

     table

     that

     sets

     me

     apart

     from

     others

     in

     my

     field

    .

     I

     am

     a

     forward

    -thinking

     thinker

    ,

     with

     a

     keen

     sense

     of

     vision

     and

     a

     passion

     for

     pushing

     the

     boundaries

     of

     what

     is

     possible

     with

     technology

    .

     I

     am

     a

     team

     player

    ,

     deeply

     committed

     to

     collaborating

     with

     others

     and

     helping

     to

     drive

     projects

     to

     their

     peak

    .

     I

     am

     also

     a

     patient

    ,

     detail

    -oriented

     individual

     who

     values

     not

     just

     technical

     knowledge

    ,

     but

     a

     deep

     understanding

     of

     the

     human

     experience

    .

     Overall

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Ex

    plain

     the

     significance

     of

     Paris

     in

     French

     culture

     and

     history

    ,

     and

     its

     current

     status

     and

     future

     prospects

    .

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     and

     it

     holds

     great

     significance

     in

     French

     culture

     and

     history

    .

     The

     city

     is

     the

     seat

     of

     government

     for

     the

     French

     Republic

    ,

     and

     it

     is

     also

     known

     as

     the

     "

    city

     of

     light

    "

     for

     its

     rich

     history

     and

     architectural

     styles

    .

     Paris

     is

     home

     to

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

     Notre

     Dame

     Cathedral

    ,

     the

     Ro

    emer

    -W

    right

     Tower

    ,

     the

     Ch

    amps

    -E

    lys

    ées

    ,

     the

     Arc

     de

     Tri

    omp

    he

    ,

     and

     many

     other

     famous

     landmarks

    .
    


    Paris

     has

     always

     been

     a

     symbol

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     continued

     advancements

     in

     technology

    ,

     increased

     integration

     with

     other

     industries

    ,

     and

     increasing

     reliance

     on

     AI

     in

     various

     applications

    .

     Here

     are

     some

     potential

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

     integration

     with

     other

     industries

    :

     AI

     will

     continue

     to

     be

     integrated

     into

     various

     sectors

    ,

     including

     healthcare

    ,

     transportation

    ,

     finance

    ,

     and

     manufacturing

    .

     This

     will

     lead

     to

     more

     efficient

     and

     effective

     operations

    ,

     as

     well

     as

     improved

     decision

    -making

    .
    


    2

    .

     Personal

    ization

     and

     automation

    :

     AI

     will

     become

     more

     personalized

    ,

     allowing

     for

     more

     accurate

     predictions

     and

     recommendations

    .

     It

     will

     also

     automate

     many

     tasks

    ,

     freeing

     up

     time

     and

     reducing

     the

     need

     for

     manual

     labor

    .
    


    3

    .

     Autonomous

     vehicles

    :

     The

    



```python
llm.shutdown()
```
