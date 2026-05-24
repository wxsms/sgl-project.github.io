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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.93it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.80it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.80it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.80it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.80it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.80it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.80it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.80it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.80it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.80it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 22.49it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:04<00:00, 22.49it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:04<00:00, 32.54it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=48.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=48.15 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=48.15 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=48.15 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=48.15 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=48.15 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=48.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=48.14 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=48.13 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=48.13 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=48.13 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=48.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=48.12 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=3584 avail_mem=48.12 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=48.11 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=48.11 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=48.11 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=48.11 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=2560 avail_mem=48.10 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=2304 avail_mem=48.10 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=2048 avail_mem=48.10 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=48.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=48.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=48.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=48.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=48.07 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s]Capturing num tokens (num_tokens=960 avail_mem=48.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=48.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s]Capturing num tokens (num_tokens=832 avail_mem=48.08 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.48it/s]Capturing num tokens (num_tokens=832 avail_mem=48.08 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.99it/s]Capturing num tokens (num_tokens=768 avail_mem=48.07 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.99it/s]Capturing num tokens (num_tokens=704 avail_mem=48.07 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.99it/s]Capturing num tokens (num_tokens=640 avail_mem=48.07 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.99it/s]Capturing num tokens (num_tokens=576 avail_mem=48.07 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.99it/s]Capturing num tokens (num_tokens=512 avail_mem=48.05 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.99it/s]Capturing num tokens (num_tokens=512 avail_mem=48.05 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=480 avail_mem=48.07 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=448 avail_mem=48.07 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=416 avail_mem=48.06 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]

    Capturing num tokens (num_tokens=384 avail_mem=48.06 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=352 avail_mem=48.06 GB):  50%|█████     | 29/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=352 avail_mem=48.06 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=320 avail_mem=48.05 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=288 avail_mem=48.05 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=256 avail_mem=48.05 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=240 avail_mem=48.04 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=224 avail_mem=48.04 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.75it/s]Capturing num tokens (num_tokens=224 avail_mem=48.04 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.80it/s]Capturing num tokens (num_tokens=208 avail_mem=48.03 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.80it/s]Capturing num tokens (num_tokens=192 avail_mem=48.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=176 avail_mem=48.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.80it/s]

    Capturing num tokens (num_tokens=160 avail_mem=48.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=144 avail_mem=48.02 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=144 avail_mem=48.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.51it/s]Capturing num tokens (num_tokens=128 avail_mem=48.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.51it/s]Capturing num tokens (num_tokens=112 avail_mem=48.02 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.51it/s]Capturing num tokens (num_tokens=96 avail_mem=48.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.51it/s] Capturing num tokens (num_tokens=80 avail_mem=48.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.51it/s]Capturing num tokens (num_tokens=64 avail_mem=48.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.51it/s]Capturing num tokens (num_tokens=64 avail_mem=48.01 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.54it/s]Capturing num tokens (num_tokens=48 avail_mem=48.00 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.54it/s]Capturing num tokens (num_tokens=32 avail_mem=48.00 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.54it/s]Capturing num tokens (num_tokens=28 avail_mem=47.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.54it/s]

    Capturing num tokens (num_tokens=24 avail_mem=47.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.54it/s]Capturing num tokens (num_tokens=20 avail_mem=47.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.54it/s]Capturing num tokens (num_tokens=20 avail_mem=47.99 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.53it/s]Capturing num tokens (num_tokens=16 avail_mem=47.99 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.53it/s]Capturing num tokens (num_tokens=12 avail_mem=47.98 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.53it/s]Capturing num tokens (num_tokens=8 avail_mem=47.98 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.53it/s] Capturing num tokens (num_tokens=4 avail_mem=47.98 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.53it/s]Capturing num tokens (num_tokens=4 avail_mem=47.98 GB): 100%|██████████| 58/58 [00:01<00:00, 41.35it/s]


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
    Generated text:  Marc and I am an avid reader, a photographer, and an avid skier. I am also a retired professional athlete and I like to share my experiences with others. This website is dedicated to sharing my experiences and the things I have learned throughout my athletic career.
    I have been a professional skier since 2010, I have been training and competing since 2011. I love skiing, and I love photography. My photography is in the style of Scott Mauk, and my writing is in the style of Neil Gaiman. I am currently a professional athlete, and I have competed in multiple winter
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the United States. He is like the boss of the whole country. He is often called the "Father of the Nation." What does the president do? Well, he decides how the country is run. He makes sure that the people who are allowed to run for president are people who do a good job. He also tries to make sure that the country is happy. He makes sure that the people are healthy and the country has enough food and water. The president also talks to the people about important things. He has to explain to the people why he made certain decisions and what he wants the people to do.
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. Brussels
    C. Antwerp
    D. Strasbourg
    Answer:
    A
    
    Question 43: In the early 20th century, the British invasion of India began in August 1903, effectively ending the British rule in India. A. Correct B. Incorrect
    Answer:
    A
    
    The greatest common divisor of 6 and 15 is ____
    A. 2
    B. 3
    C. 6
    D. 15
    Answer:
    A
    
    The key points to be aware of when using the 'Work Sheet for Batch Report'
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be more precise and personal, writes Garry P. Dunn, Associate Director for Research at the Software Institute. He discusses a new article by Ian Goodfellow, which shows that human-like AI can be built using deep neural networks with hundreds of layers and millions of parameters. It’s the culmination of a decade-long effort by researchers, including the University of Toronto and DeepMind.
    The ability to build a human-like AI is a dream for many, but it’s only now becoming a reality. In this talk, I discuss how the past decade of deep learning has led to an unprecedented surge in the sophistication of the algorithms used to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your profession or experience here]. I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What are some of your favorite things to do? I love [insert a short description of your favorite activities or hobbies here]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a major tourist destination. The city is also known for its cuisine, including French cuisine, which is famous for its rich flavors and use of herbs and spices. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique and fascinating city. The city is also home to many international organizations and institutions
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the potential for AI to be used for malicious purposes.
    
    2. Greater integration with human decision-making: AI is likely to become more integrated with human decision-making in the future, as AI systems are expected to be able to make decisions
    


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
    Generated text:  [Your Name] and I am a [job title] at [company name]. I have over [number of years] years of experience in [industry], and have a passion for [main interest/interests]. I am confident in my abilities and always aim to exceed expectations, both in terms of technical skills and interpersonal skills. I am a team player, reliable, and detail-oriented. I enjoy sharing my knowledge and experience with others, and I strive to learn and improve continually. I am always eager to challenge myself and push beyond my comfort zone. I am a [positive trait] person who always strives to be [desired outcome
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Based on the given information, what is the capital city of France?
    
    The capital city of France is Paris. 
    
    To verify this, I used the following knowledge point: The capital city of France is Paris. This answer directly addresses the statement provided and provides the correct information. 
    
    I do not need to create a new fact, but I can explain the reasoning behind my answer:
    
    1. The question asks for the capital city of France, which is explicitly stated in the information provided.
    2. The capital cities of France are determined by their administrative status, with the most populous city being the capital.
    3. Paris is the capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but there are several potential trends that may shape the technology's direction in the coming years. Here are a few of the most likely ones:
    
    1. Increased automation: As AI technology continues to advance, it's likely that we'll see more widespread automation of routine tasks, such as data entry, administrative work, and repetitive labor. This could lead to increased efficiency and cost savings for businesses.
    
    2. AI personalization: AI is already becoming more personalized, with the ability to learn from user data and make personalized recommendations. As AI technology continues to improve, we may see even more personalized experiences in AI-driven products and services.
    
    3


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

    ...

     what

    ?

     Are

     you

     related

     to

     the

     main

     character

     in

     the

     story

     I

    'm

     writing

    ?

     If

     so

    ,

     please

     provide

     details

     about

     your

     relationship

    .

     Alternatively

    ,

     you

     could

     answer

     "

    Un

    related

    "

     if

     you

     are

     not

     related

     to

     the

     main

     character

    .

     If

     you

     don

    't

     have

     any

     details

     about

     your

     relationship

    ,

     you

     can

     just

     say

     "

    I

    'm

     just

     a

     regular

     person

    ".

     Good

     luck

     with

     the

     story

    !

     Let

     me

     know

     if

     you

     have

     any

     other

     questions

    .

     [

    Insert

     your

     name

     here

    ]

     How

     are

     you

     doing

     today

    ?

     I

    'm

     just

     a

     regular

     person

    ,

     but

     I

    'm

     excited

     to

     get

     started

     on

     my

     next

     story

    !

     I

    'm

     glad

     you

    're

     reading

     it

    .

     How

     about

     you

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Task

    :

     Craft

     a

     short

     story

     with

     the

     following

     elements

    :

     Subject

     (

    a

     person

    's

     name

    ),

     setting

    ,

     problem

    ,

     solution

    ,

     and

     theme

     (

    what

     the

     story

     explores

    ).

     


    In

     a

     small

     town

     called

     B

    akers

    field

    ,

     a

     woman

     named

     Laura

     found

     herself

     in

     a

     predic

    ament

    .

     She

     had

     recently

     lost

     her

     job

     and

     was

     struggling

     to

     keep

     up

     with

     her

     bills

     and

     pay

    .

     Laura

    's

     best

     friend

    ,

     Emily

    ,

     recommended

     a

     friend

    ,

     Sarah

    ,

     as

     the

     perfect

     roommate

    .

     Sarah

     was

     kind

    ,

     honest

    ,

     and

     experienced

    .

     Laura

     was

     hesitant

     to

     share

     her

     problems

     with

     her

     new

     roommate

    ,

     but

     she

     decided

     to

     try

     and

     help

    .


    Sarah

     was

     a

     talented

     musician

     who

     lived

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     very

     exciting

    ,

     with

     significant

     advancements

     on

     various

     fronts

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Depend

    ence

     on

     AI

     for

     Decision

    -M

    aking

    :

     With

     the

     increasing

     reliance

     on

     AI

     for

     decision

    -making

     in

     various

     fields

    ,

     it

     is

     likely

     that

     we

     will

     see

     more

     reliance

     on

     AI

     for

     critical

     decisions

    ,

     such

     as

     in

     healthcare

    ,

     finance

    ,

     and

     military

     operations

    .
    


    2

    .

     AI

     Personal

    ization

    :

     With

     the

     rise

     of

     big

     data

     and

     machine

     learning

    ,

     it

     is

     likely

     that

     we

     will

     see

     a

     continued

     focus

     on

     personal

    izing

     the

     AI

     systems

     that

     we

     use

    .

     This

     could

     include

     things

     like

     personalized

     medicine

    ,

     chat

    bots

    ,

     and

     virtual

     assistants

    .
    


    3

    .

     AI

     in

    



```python
llm.shutdown()
```
