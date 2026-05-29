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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:20,  4.57s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.24it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.52it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.39it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.37it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.26it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.14it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.14it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.14it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.44it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.11it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.64it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.64it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.64it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.64it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.64it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.64it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.56it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.56it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.56it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.56it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.56it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.56it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.56it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.28it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.28it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.28it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.28it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.28it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.28it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=208 avail_mem=74.57 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.57 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=176 avail_mem=74.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=160 avail_mem=74.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=144 avail_mem=74.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=144 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.16it/s]Capturing num tokens (num_tokens=128 avail_mem=74.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.16it/s]Capturing num tokens (num_tokens=112 avail_mem=73.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.16it/s]Capturing num tokens (num_tokens=96 avail_mem=73.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.16it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=73.61 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.16it/s]Capturing num tokens (num_tokens=64 avail_mem=73.61 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.16it/s]Capturing num tokens (num_tokens=64 avail_mem=73.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.57it/s]Capturing num tokens (num_tokens=48 avail_mem=73.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.57it/s]Capturing num tokens (num_tokens=32 avail_mem=73.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.57it/s]Capturing num tokens (num_tokens=28 avail_mem=73.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.57it/s]Capturing num tokens (num_tokens=24 avail_mem=73.59 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.57it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.57it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=16 avail_mem=73.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.12it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=73.58 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:01<00:00, 38.69it/s]


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
    Generated text:  Amelia and I am from the United States. I am a writer and I am here to teach you a few words that you should know. But first, here are some words that you should know.
    In the world of the English language, there are so many words that you should know. Every word has its own meaning. Some of the most common words that you should know are:
    1. Word: a phrase or word that is composed of letters and sounds. For example, "apple" is a word that can be used to describe an apple, a type of fruit.
    2. Verb: a word that expresses a state of being
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He is called the President of the United States. When he takes office, he becomes the President of the United States, and he is the president of a country. The president is the president of the United States. The president of the United States is a very important person in the country. He is called the President of the United States. When he takes office, he becomes the President of the United States, and he is the president of a country. The president of the United States is a very important person in the country. He is called the President of the United States. When he takes office
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. Correct
    B. Incorrect
    Answer: A
    
    The duration of the probationary period for Party members is the same as that of formal members, and the probationary period is calculated from the date of the Party committee's decision on its acceptance.
    A. Correct
    B. Incorrect
    Answer: A
    
    In the event of a fire at the station, the on-duty station manager must immediately ___ and report the situation to the control center.
    A. Evacuate the station
    B. Call 119
    C. Initiate the fire mode
    D. Organize the crowd evacuation
    Answer: D
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly changing, with the rise of new technologies and emerging issues. One of the most significant advancements in AI is the development of natural language processing (NLP) technology. NLP has the potential to revolutionize the way we communicate and understand language, and it is shaping up to be a major force in the future of AI.
    
    NLP is a branch of artificial intelligence that focuses on the ability of machines to understand, interpret, generate, and generate human language. NLP has become increasingly important in recent years as it has been applied to a wide range of fields, including healthcare, legal, and social media.
    
    One of the key aspects


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


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose" (the Rose City). It is the largest city in France and the third largest in the world, with a population of over 2 million people. Paris is a cultural and historical center, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major financial center and a major tourist destination. Paris is a city of contrasts, with its rich history and modernity blending together to create a unique and fascinating city. The city is home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs and preferences.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be an increased need for privacy and security measures to protect the data and information that is generated and processed by
    


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
    Generated text:  [Name]. I'm a [Age] year old [Country] [Occupation]. I'm here to share my story with you.
    
    As a human being, I am constantly seeking to understand the world around me and the challenges that I face. I hope that I can connect with others and contribute my knowledge and experience to make the world a better place. Thank you for the opportunity to meet you.
    
    Question: What is your profession? How did you become a professional in this field? I'm an [Occupation] and I have [X number of years] of experience in this field.
    
    What is your favorite hobby? It's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Your response should be around 50 words. 
    
    Paris is the largest city in France and is the capital of the country. It is located on the right bank of the Seine River. The city is home to many museums, such as the Louvre and Musée d'Orsay. Paris is also known for its food, culture, and architecture, including its iconic Notre-Dame Cathedral. It is an international financial and trade center, and has a rich cultural heritage. The city is home to many museums, art galleries, and historical sites that attract millions of visitors each year. 
    
    Paris is a global city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undoubtedly fascinating and unpredictable, and it is challenging to predict its exact trajectory. However, there are several trends that are likely to shape the AI landscape in the coming years. Here are some of the most notable:
    
    1. Increased focus on ethical AI: AI is increasingly being used in fields such as healthcare, finance, and transportation, but ethical concerns are also becoming increasingly important. Governments and industry leaders are starting to take a more proactive approach to designing AI systems that are fair, transparent, and accountable.
    
    2. Greater adoption of AI in natural language processing: AI is increasingly being used in natural language processing (NLP), which refers to the


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

    ]

     and

     I

     am

     [

    Your

     Age

    ].

     I

     am

     an

     [

    Type

     of

     Job

    ],

     [

    Your

     Field

     of

     Expert

    ise

    ],

     [

    Your

     Expert

    ise

    ],

     and

     [

    Your

     Skills

    ]

     [

    Your

     Expert

    ise

    ].

     I

     have

     worked

     in

     [

    Your

     Role

    ],

     [

    Your

     Company

    ],

     [

    Your

     Organization

    ],

     [

    Your

     Team

    ],

     and

     [

    Your

     Department

    ].

     I

     also

     hold

     [

    Your

     Degree

     or

     Certification

    ],

     [

    Your

     Experience

     Level

    ],

     and

     [

    Your

     Leadership

     Skills

    ].

     I

     enjoy

     [

    Your

     Inter

    ests

     or

     Passion

    ],

     and

     I

     am

     always

     looking

     for

     ways

     to

     grow

     and

     learn

    .

     I

     am

     eager

     to

     get

     to

     know

     you

     and

     help

     you

     achieve

     your

     goals

    .

     How

     can

     I

     assist

     you

     today

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Its

     name

     means

     "

    city

     of

     light

    "

     in

     French

    .

     It

     is

     located

     in

     the

     Lo

    ire

     Valley

     region

     of

     France

    ,

     on

     the

     banks

     of

     the

     Gar

    onne

     River

    ,

     and

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     country

    .

     Paris

     is

     the

     world

    's

     

    1

    1

    th

    -largest

     city

     and

     the

     

    1

    2

    th

    -largest

     metropolitan

     area

    .

     It

     is

     home

     to

     UNESCO

     World

     Heritage

     sites

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     is

     also

     the

     home

     of

     the

     French

     President

    ,

     the

     French

     parliament

    ,

     and

     the

     French

     Parliament

     building

    .

     It

     is

     the

     cultural

    ,

     economic

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

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

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     will

     likely

     become

     more

     integrated

     with

     human

     intelligence

     to

     better

     understand

     and

     respond

     to

     human

     needs

    .

     This

     may

     lead

     to

     new

     ways

     of

     using

     AI

     to

     enhance

     human

     capabilities

    ,

     such

     as

     autonomous

     vehicles

     or

     virtual

     assistants

     designed

     to

     assist

     with

     daily

     tasks

    .
    


    2

    .

     Adv

    ancements

     in

     language

     and

     semantics

    :

     AI

     systems

     will

     continue

     to

     improve

     their

     ability

     to

     understand

     and

     generate

     human

     language

    ,

     leading

     to

     new

     applications

     in

     fields

     such

     as

     education

    ,

     language

     translation

    ,

     and

     virtual

     assistants

    .
    


    3

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

    



```python
llm.shutdown()
```
