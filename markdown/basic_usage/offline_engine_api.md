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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:38,  3.83s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:38,  3.83s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:38,  3.83s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:38,  3.83s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:38,  3.83s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:31,  1.69it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:31,  1.69it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:31,  1.69it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:31,  1.69it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:31,  1.69it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.69it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:31,  1.69it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:31,  1.69it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  5.00it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 11.12it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 11.12it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.70it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.70it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.70it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.70it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.70it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.70it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.70it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.70it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.70it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.70it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.70it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 26.45it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 26.45it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 26.45it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 26.45it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 26.45it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 26.45it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 26.45it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 26.45it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 26.45it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 26.45it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 26.45it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.89it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.89it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.89it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.89it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.89it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.89it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.89it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=49.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=49.75 GB):   3%|▎         | 2/58 [00:00<00:02, 19.31it/s]Capturing num tokens (num_tokens=6656 avail_mem=49.75 GB):   3%|▎         | 2/58 [00:00<00:02, 19.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=49.75 GB):   3%|▎         | 2/58 [00:00<00:02, 19.31it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=49.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=49.74 GB):   9%|▊         | 5/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=5120 avail_mem=49.73 GB):   9%|▊         | 5/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=4608 avail_mem=49.73 GB):   9%|▊         | 5/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=49.73 GB):   9%|▊         | 5/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=49.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]Capturing num tokens (num_tokens=3840 avail_mem=49.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]Capturing num tokens (num_tokens=3584 avail_mem=49.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=49.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]Capturing num tokens (num_tokens=3072 avail_mem=49.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=49.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.31it/s]Capturing num tokens (num_tokens=2816 avail_mem=49.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=49.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=49.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=49.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=49.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=49.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=49.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=49.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=1024 avail_mem=49.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=960 avail_mem=49.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s] Capturing num tokens (num_tokens=896 avail_mem=49.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]

    Capturing num tokens (num_tokens=832 avail_mem=49.68 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.84it/s]Capturing num tokens (num_tokens=832 avail_mem=49.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=768 avail_mem=49.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=704 avail_mem=49.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=640 avail_mem=49.67 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=576 avail_mem=49.67 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=512 avail_mem=49.66 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=512 avail_mem=49.66 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]Capturing num tokens (num_tokens=480 avail_mem=49.67 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]Capturing num tokens (num_tokens=448 avail_mem=49.67 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]Capturing num tokens (num_tokens=416 avail_mem=49.67 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]Capturing num tokens (num_tokens=384 avail_mem=49.67 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]

    Capturing num tokens (num_tokens=352 avail_mem=49.66 GB):  50%|█████     | 29/58 [00:00<00:00, 42.71it/s]Capturing num tokens (num_tokens=352 avail_mem=49.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=320 avail_mem=49.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=288 avail_mem=49.65 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=256 avail_mem=49.65 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=240 avail_mem=49.65 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=224 avail_mem=49.65 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.43it/s]Capturing num tokens (num_tokens=224 avail_mem=49.65 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.73it/s]Capturing num tokens (num_tokens=208 avail_mem=49.64 GB):  67%|██████▋   | 39/58 [00:00<00:00, 45.73it/s]Capturing num tokens (num_tokens=192 avail_mem=49.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=176 avail_mem=49.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=160 avail_mem=49.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.73it/s]

    Capturing num tokens (num_tokens=144 avail_mem=49.63 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=144 avail_mem=49.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=128 avail_mem=49.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=112 avail_mem=49.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=96 avail_mem=49.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s] Capturing num tokens (num_tokens=80 avail_mem=49.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=64 avail_mem=49.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.58it/s]Capturing num tokens (num_tokens=64 avail_mem=49.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=48 avail_mem=49.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=32 avail_mem=49.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=28 avail_mem=49.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=24 avail_mem=49.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]

    Capturing num tokens (num_tokens=20 avail_mem=49.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=20 avail_mem=49.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=16 avail_mem=49.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=12 avail_mem=49.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=8 avail_mem=49.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.82it/s] Capturing num tokens (num_tokens=4 avail_mem=49.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=4 avail_mem=49.59 GB): 100%|██████████| 58/58 [00:01<00:00, 41.59it/s]


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
    Generated text:  Adrian, and I’m a 14 year old boy from my home in the northern part of the United States. I lived here since I was young and I have loved it since I could remember. My family and I live in a three-story house with 13 rooms. I love to read, go on adventures with my family, ride my bicycle around town, and attend school. I have also participated in some sports and I have won many trophies in them. What are the most interesting things about your home? 1. The pictures that show you living there. 2. The details of the house. 3.
    ===============================
    Prompt: The president of the United States is
    Generated text:  34 years older than the president of巴西, and the president of the United States is 70 years older than the president of Peru. If the president of Peru is 30 years old, what is the sum of the ages of all three presidents?
    To determine the sum of the ages of the presidents of the United States, Brazil, and Peru, we need to follow these steps:
    
    1. Identify the age of the president of Peru.
    2. Use the age of Peru to find the age of the president of Brazil.
    3. Use the age of Brazil to find the age of the president of the United States.
    
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    B) Lyon
    C) Marseille
    D) Brussels
    E) Nice
    
    To determine the capital of France, let's analyze the options step by step:
    
    1. **Paris**: This is the capital of France, located in the center of the country.
    2. **Lyon**: This is a city in the Rhône Valley, not the capital of France.
    3. **Marseille**: This is a city in the south of France, not the capital of France.
    4. **Brussels**: This is a capital city of Belgium, not of France.
    5. **Nice**: This is a
    ===============================
    Prompt: The future of AI is
    Generated text:  promising, and it is predicted that by 2025, AI will represent around 16% of global GDP, while 40% of global internet traffic is expected to be generated by AI. The future of AI is likely to revolutionize industries across various sectors, including healthcare, finance, and transportation, among others. AI is already being used to improve patient outcomes, optimize supply chains, and enhance cybersecurity. AI is also being used to predict market trends, automate repetitive tasks, and provide more accurate health diagnoses. As AI technology continues to evolve, it will become even more personalized and adaptable, allowing for a more seamless


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am a [job title] at [company name] and I have been with the company for [number of years] years. I have always been passionate about [job title] and have always wanted to do what I do. I am a [job title] at [company name] and I have always been passionate about [job title] and have always wanted to do what I do. I am a [job title] at [company name] and I have always been passionate about [job title] and have always wanted to do what I do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is the capital of France and the largest city in the European Union. It is also the birthplace of the French Revolution and the French Revolution is considered one of the most significant events in French history. Paris is a city of contrasts, with its modern architecture and historical landmarks, and is a popular destination for tourists and locals
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives through the use of voice assistants like Siri and Alexa, smart home devices, and self-driving cars. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption of AI in our everyday activities.
    
    2. AI will become more autonomous: As AI becomes more integrated into our daily lives, we can expect to see more
    


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
    Generated text:  [insert your name] and I am a [insert your profession or role]. I am passionate about [insert a personal interest or hobby]. How are you today? As a [insert your profession or role], I am always here to assist you with whatever you need. What can I do for you today? I love to help people and find solutions to their problems. How can I help you today? And what would you like to know or discuss? I look forward to working with you. Please feel free to ask me any questions or share any information you have. Thank you for having me. 
    (Note: You can replace [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest and most populous city in France, with an estimated population of over 2. 5 million people. The city is known for its beautiful architecture, rich history, and vibrant culture. It is home to iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also the world's most visited city, attracting millions of visitors each year. The city has played a major role in French history, including the French Revolution and Napoleon's conquest of France. Today, Paris remains one of the world's top cities for education, entertainment, and shopping.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, with a range of new technologies and trends emerging that could significantly impact how we live, work, and interact with technology.
    
    One of the key trends in AI is the development of more advanced natural language processing and machine learning algorithms. This could lead to more accurate and sophisticated speech recognition and text-to-speech technology, as well as improved ability to understand and respond to complex human language.
    
    Another trend is the increased use of AI in areas such as healthcare, where AI-powered diagnostic tools and personalized treatment plans could improve patient outcomes. Additionally, AI could play a more significant role in renewable energy technologies, such as the use of AI to


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

     Emily

    .

     I

     am

     an

     adult

     with

     about

     

    3

    0

     years

     of

     experience

     in

     the

     tech

     industry

    .

     I

     have

     always

     been

     fascinated

     by

     the

     world

     of

     technology

     and

     have

     always

     wanted

     to

     learn

     more

     about

     it

    .

     I

     have

     a

     strong

     interest

     in

     programming

     and

     have

     been

     working

     on

     a

     project

     for

     the

     past

     few

     years

    .

     I

     am

     a

     quiet

     and

     serious

     person

     who

     values

     hard

     work

     and

     dedication

    .

     I

     am

     always

     eager

     to

     learn

     and

     seek

     out

     new

     challenges

    .

     I

     am

     always

     looking

     for

     new

     ways

     to

     enhance

     my

     skills

     and

     am

     always

     willing

     to

     learn

     and

     grow

    .
    


    Can

     you

     tell

     me

     more

     about

     your

     project

     and

     what

     you

     hope

     to

     achieve

     with

     it

    ?

     Emily

    ,

     your

     project

     is

     not

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     the

     

    8

    th

    -largest

     city

     in

     the

     world

    ,

     is

     France

    's

     largest

     and

     most

     populous

     city

    ,

     and

     is

     also

     the

     capital

     city

     of

     the

     country

    .

     The

     city

     was

     founded

     by

     the

     Romans

     and

     was

     the

     capital

     of

     the

     Ancient

     Gaul

    s

    ,

     then

     of

     the

     Roman

     Empire

    ,

     the

     Carol

    ing

    ian

     Empire

    ,

     the

     Ange

    vin

     Empire

    ,

     and

     the

     French

     Empire

    .

     Paris

     is

     renowned

     for

     its

     many

     museums

    ,

     a

     rich

     cultural

     life

    ,

     and

     its

     various

     historical

     monuments

    ,

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

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

     The

     city

     is

     also

     known

     for

     its

     fashion

     industry

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

     and

     there

     are

     many

     possible

     trends

     that

     could

     shape

     the

     way

     it

     is

     developed

    ,

     used

    ,

     and

     deployed

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     that

     could

     be

     expected

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     accessibility

    :

     AI

     will

     become

     more

     accessible

     to

     a

     wider

     audience

    ,

     enabling

     people

     to

     use

     it

     in

     new

     ways

     and

     to

     solve

     problems

     that

     were

     previously

     out

     of

     reach

    .

     This

     will

     require

     improvements

     in

     technology

    ,

     data

     privacy

    ,

     and

     ethical

     considerations

    .
    


    2

    .

     Autonomous

     machines

    :

     AI

     will

     continue

     to

     evolve

     towards

     more

     autonomous

     machines

     that

     can

     perform

     tasks

     without

     human

     intervention

    .

     This

     could

     lead

     to

     more

     efficient

     production

    ,

     improved

     healthcare

     outcomes

    ,

     and

     reduced

     human

     error

    .
    


    3

    .

    



```python
llm.shutdown()
```
