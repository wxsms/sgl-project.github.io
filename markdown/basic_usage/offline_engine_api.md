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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.97it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:04<00:14,  3.37it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:05,  7.92it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:05,  7.92it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:05,  7.92it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:05,  7.92it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:05,  7.92it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:04<00:05,  7.92it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:04<00:05,  7.92it/s]Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:04<00:05,  7.92it/s]Compiling num tokens (num_tokens=768):  29%|██▉       | 17/58 [00:04<00:05,  7.92it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=352):  43%|████▎     | 25/58 [00:04<00:02, 13.53it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:04<00:01, 21.12it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:04<00:01, 21.12it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:04<00:01, 21.12it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:04<00:01, 21.12it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:04<00:01, 21.12it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:04<00:01, 21.12it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:04<00:01, 21.12it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:04<00:01, 21.12it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 27.15it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 37.79it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 37.79it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 37.79it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 37.79it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 37.79it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 37.79it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 37.79it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 37.79it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.20 GB):   3%|▎         | 2/58 [00:00<00:02, 19.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.20 GB):   3%|▎         | 2/58 [00:00<00:02, 19.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.20 GB):   3%|▎         | 2/58 [00:00<00:02, 19.47it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.20 GB):   3%|▎         | 2/58 [00:00<00:02, 19.47it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.20 GB):   9%|▊         | 5/58 [00:00<00:02, 22.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.18 GB):   9%|▊         | 5/58 [00:00<00:02, 22.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.18 GB):   9%|▊         | 5/58 [00:00<00:02, 22.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.18 GB):   9%|▊         | 5/58 [00:00<00:02, 22.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.52it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.52it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.52it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.15 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.15 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.14 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.83it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.83it/s]Capturing num tokens (num_tokens=960 avail_mem=74.14 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.83it/s] Capturing num tokens (num_tokens=896 avail_mem=74.14 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.83it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.13 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.83it/s]Capturing num tokens (num_tokens=768 avail_mem=74.13 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.83it/s]Capturing num tokens (num_tokens=768 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=704 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=640 avail_mem=74.12 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=576 avail_mem=74.12 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=512 avail_mem=74.11 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=480 avail_mem=74.12 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=448 avail_mem=74.12 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=448 avail_mem=74.12 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.51it/s]Capturing num tokens (num_tokens=416 avail_mem=74.12 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.51it/s]Capturing num tokens (num_tokens=384 avail_mem=74.12 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.51it/s]Capturing num tokens (num_tokens=352 avail_mem=74.11 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.51it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.10 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.51it/s]Capturing num tokens (num_tokens=288 avail_mem=74.10 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.51it/s]Capturing num tokens (num_tokens=288 avail_mem=74.10 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.80it/s]Capturing num tokens (num_tokens=256 avail_mem=74.10 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.80it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.80it/s]Capturing num tokens (num_tokens=224 avail_mem=74.09 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.80it/s]Capturing num tokens (num_tokens=208 avail_mem=74.09 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.80it/s]Capturing num tokens (num_tokens=192 avail_mem=74.09 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.80it/s]Capturing num tokens (num_tokens=176 avail_mem=74.08 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.80it/s]Capturing num tokens (num_tokens=176 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.24it/s]Capturing num tokens (num_tokens=160 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.24it/s]Capturing num tokens (num_tokens=144 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.24it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.24it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.24it/s]Capturing num tokens (num_tokens=96 avail_mem=74.07 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.24it/s] Capturing num tokens (num_tokens=96 avail_mem=74.07 GB):  81%|████████  | 47/58 [00:01<00:00, 48.53it/s]Capturing num tokens (num_tokens=80 avail_mem=74.07 GB):  81%|████████  | 47/58 [00:01<00:00, 48.53it/s]Capturing num tokens (num_tokens=64 avail_mem=74.06 GB):  81%|████████  | 47/58 [00:01<00:00, 48.53it/s]Capturing num tokens (num_tokens=48 avail_mem=74.06 GB):  81%|████████  | 47/58 [00:01<00:00, 48.53it/s]Capturing num tokens (num_tokens=32 avail_mem=74.06 GB):  81%|████████  | 47/58 [00:01<00:00, 48.53it/s]Capturing num tokens (num_tokens=28 avail_mem=74.05 GB):  81%|████████  | 47/58 [00:01<00:00, 48.53it/s]Capturing num tokens (num_tokens=28 avail_mem=74.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.82it/s]Capturing num tokens (num_tokens=24 avail_mem=74.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.82it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.82it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.82it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.04 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.82it/s]Capturing num tokens (num_tokens=8 avail_mem=74.04 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.82it/s] Capturing num tokens (num_tokens=4 avail_mem=74.03 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.82it/s]Capturing num tokens (num_tokens=4 avail_mem=74.03 GB): 100%|██████████| 58/58 [00:01<00:00, 49.38it/s]Capturing num tokens (num_tokens=4 avail_mem=74.03 GB): 100%|██████████| 58/58 [00:01<00:00, 43.34it/s]


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
    Generated text:  Jimmy and I'm a great amateur writer in the games genre, but I've been told that when I'm writing and editing a book, I can't tell it's a book, because it's not. How do I fix that? I'm still a writer and I'm very much in love with writing. I have no interest in this other side of my career. I just want to produce a good book. When I'm writing the book, it's always for fun and I'm a writer, but I'm not too interested in this side of writing. 
    
    I've tried reading books I like by others and they're full
    ===============================
    Prompt: The president of the United States is
    Generated text:  a position he has held since 1876, and this has led him to be included in the United States Senate. What is the current term for this position? The current term for the president of the United States is 4 years. This term begins on January 20, 2023, and ends on January 20, 2027. The president is responsible for overseeing the executive branch, including the management of the federal budget, the defense, foreign policy, and the national security apparatus. He also has the power to appoint federal judges, ambassadors, and other federal officers. The president
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. London
    C. Madrid
    D. Rome
    Answer: A
    
    The capital of the Qing Dynasty was _______.
    A. Beijing
    B. Xi'an
    C. Shanghai
    D. Guangzhou
    Answer: A
    
    Which of the following is a principle of work safety?
    A. No one should be allowed to operate equipment beyond their own skill level.
    B. Safety first, prevention foremost.
    C. Management should be responsible.
    D. People-oriented.
    Answer: B
    
    The capital of Germany is ________.
    A. Berlin
    B. Munich
    C. Frankfurt
    D.
    ===============================
    Prompt: The future of AI is
    Generated text:  an exciting topic. While we can only describe a few of the advancements that will be made in the near future, one thing is certain - the AI of the future will not be limited by today's technology. With a continuing increase in computing power and increased access to the internet, we can expect more AI to emerge and become more powerful.
    One of the most promising areas of AI development is in the field of natural language processing. This is the ability of AI systems to understand and process human language, which includes natural language processing. This includes things like sentiment analysis, language translation, and speech recognition. With the continued growth of natural language processing


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the arts and sciences. It is also home to the world's oldest university, the University of Paris. Paris is a popular tourist destination, attracting millions of visitors each year. It is also a major center for the French language and culture, with many French restaurants, cafes, and shops. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more robust AI systems that are designed to be
    


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
    Generated text:  [Character Name]. I'm an artist who has been inspired by the beauty of nature for as long as I can remember. My work is inspired by the vastness of space and the beauty of the universe, and I'm always striving to capture the essence of the natural world through my art. I believe that art is a powerful tool for exploring the depths of our minds and for bringing beauty and wonder to our world. I'm a believer in the power of interconnectedness and the importance of protecting our planet, and I want to continue to inspire and empower others to do the same. So if you're interested in learning more about my work
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe by area, with a population of over 2. 5 million people.
    
    That's great! Could you tell me more about the cultural significance of Paris? Sure! Paris is one of the most important cultural cities in the world, with a rich history and a vibrant arts scene. The city is home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It is also known for its jazz and art scenes, with numerous museums, galleries, and theaters. Paris is also known for its fashion industry, with numerous fashion houses and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dynamic and is likely to involve a combination of disruptive innovations, continuous improvements, and continued exploration of new areas. Here are some potential trends that may emerge in the years ahead:
    
    1. Increased focus on ethical considerations: As AI is increasingly integrated into various industries, there is a growing emphasis on ethical considerations, privacy, and accountability. This may lead to stricter regulations and greater transparency in the use of AI technologies.
    2. Advancements in quantum computing: Quantum computing has the potential to revolutionize AI by enabling faster and more powerful algorithms. This could lead to significant advancements in AI applications, such as predictive analytics, drug discovery, and algorithm


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

     at

     [

    company

     name

    ].

     I

    'm

     passionate

     about

     [

    insert

     what

     you

    're

     passionate

     about

    ]

     and

     always

     aim

     to

     make

     a

     positive

     impact

     on

     the

     world

    .

     I

    'm

     a

     team

     player

     and

     thrive

     on

     collaboration

     and

     teamwork

    .

     What

     kind

     of

     experience

     would

     you

     like

     to

     share

     with

     us

    ,

     and

     how

     can

     we

     arrange

     a

     conversation

     about

     it

    ?

     [

    Name

    ]

     approaches

     the

     conversation

     with

     enthusiasm

     and

     a

     willingness

     to

     learn

    .

     No

     need

     to

     re

    hash

     the

     same

     lines

     about

     the

     company

    's

     mission

     and

     goals

    .

     Simply

     state

     what

     you

     want

     to

     share

     and

     how

     you

     can

     help

    .

     I

     look

     forward

     to

     our

     discussion

     and

     see

     where

     this

     conversation

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     region

     of

     the

     same

     name

    ,

     and

     it

     is

     the

     largest

     city

     in

     Europe

     by

     population

    .

     Paris

     was

     founded

     in

     the

     

    1

    2

    th

     century

     as

     a

     military

     outpost

     and

     later

     became

     a

     major

     cultural

     and

     economic

     center

    .

     It

     is

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

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     known

     for

     its

     rich

     history

    ,

     with

     many

     significant

     historical

     sites

    ,

     museums

    ,

     and

     monuments

    .

     The

     city

     has

     a

     multicultural

     population

     and

     is

     known

     for

     its

     gastr

    onomy

    ,

     art

    ,

     and

     fashion

    .

     Paris

     has

     a

     rich

     tradition

     of

     literature

    ,

     with

     numerous

     literary

     festivals

     and

     literary

     events

     taking

     place

     throughout

     the

     year

    .

     It

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

     and

     there

     are

     several

     potential

     trends

     that

     are

     likely

     to

     shape

     its

     development

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

     automation

    :

     AI

     will

     continue

     to

     become

     more

     advanced

     and

     efficient

    ,

     enabling

     machines

     to

     perform

     a

     wide

     range

     of

     tasks

    ,

     from

     data

     analysis

     and

     decision

    -making

     to

     routine

     maintenance

     and

     repair

    .
    


    2

    .

     Increased

     reliance

     on

     machine

     learning

    :

     More

     and

     more

     AI

     systems

     will

     rely

     on

     machine

     learning

     algorithms

    ,

     which

     allow

     machines

     to

     learn

     from

     data

     and

     make

     predictions

     or

     decisions

     without

     being

     explicitly

     programmed

    .
    


    3

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     will

     become

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     bi

    otech

     and

     renewable

     energy

    ,

     to

     create

     smarter

    ,

    



```python
llm.shutdown()
```
