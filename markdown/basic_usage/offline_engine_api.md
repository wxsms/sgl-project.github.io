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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.29it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:33,  4.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:33,  4.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:33,  4.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:33,  4.81s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:33,  4.81s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:12,  3.64it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.19it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.91it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.91it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.91it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.91it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.91it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.91it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.91it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.91it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.91it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.91it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:05<00:02, 13.91it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.65it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.65it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.65it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.65it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.65it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.65it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.65it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.65it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.65it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.65it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 21.65it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.97 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:03, 18.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:03, 18.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:03, 18.56it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:03, 18.56it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.94 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.93 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.92 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.92 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.92 GB):   9%|▊         | 5/58 [00:00<00:02, 21.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.92 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.91 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.91 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.91 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.90 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.90 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.89 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.89 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.89 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.88 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.86 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.56it/s]Capturing num tokens (num_tokens=960 avail_mem=72.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.56it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.87 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.56it/s]Capturing num tokens (num_tokens=832 avail_mem=72.87 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.56it/s]Capturing num tokens (num_tokens=832 avail_mem=72.87 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=768 avail_mem=72.87 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=704 avail_mem=72.86 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=640 avail_mem=72.86 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=576 avail_mem=72.86 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=512 avail_mem=72.84 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=512 avail_mem=72.84 GB):  50%|█████     | 29/58 [00:00<00:00, 43.97it/s]Capturing num tokens (num_tokens=480 avail_mem=72.86 GB):  50%|█████     | 29/58 [00:00<00:00, 43.97it/s]Capturing num tokens (num_tokens=448 avail_mem=72.86 GB):  50%|█████     | 29/58 [00:00<00:00, 43.97it/s]Capturing num tokens (num_tokens=416 avail_mem=72.86 GB):  50%|█████     | 29/58 [00:00<00:00, 43.97it/s]Capturing num tokens (num_tokens=384 avail_mem=72.85 GB):  50%|█████     | 29/58 [00:00<00:00, 43.97it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.85 GB):  50%|█████     | 29/58 [00:00<00:00, 43.97it/s]Capturing num tokens (num_tokens=320 avail_mem=72.84 GB):  50%|█████     | 29/58 [00:00<00:00, 43.97it/s]Capturing num tokens (num_tokens=320 avail_mem=72.84 GB):  60%|██████    | 35/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=288 avail_mem=72.84 GB):  60%|██████    | 35/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=256 avail_mem=72.84 GB):  60%|██████    | 35/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=240 avail_mem=72.83 GB):  60%|██████    | 35/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=224 avail_mem=72.83 GB):  60%|██████    | 35/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=208 avail_mem=72.83 GB):  60%|██████    | 35/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=208 avail_mem=72.83 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=192 avail_mem=72.83 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=176 avail_mem=72.82 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=160 avail_mem=72.82 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.47it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.82 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=128 avail_mem=72.82 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=128 avail_mem=72.82 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.69it/s]Capturing num tokens (num_tokens=112 avail_mem=72.81 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.69it/s]Capturing num tokens (num_tokens=96 avail_mem=72.81 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.69it/s] Capturing num tokens (num_tokens=80 avail_mem=72.81 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.69it/s]Capturing num tokens (num_tokens=64 avail_mem=72.80 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.69it/s]Capturing num tokens (num_tokens=48 avail_mem=72.80 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.69it/s]Capturing num tokens (num_tokens=48 avail_mem=72.80 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=32 avail_mem=72.80 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=28 avail_mem=72.79 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=24 avail_mem=72.79 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.97it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.78 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=16 avail_mem=72.78 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=16 avail_mem=72.78 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=12 avail_mem=72.78 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=8 avail_mem=72.78 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.81it/s] Capturing num tokens (num_tokens=4 avail_mem=72.77 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=4 avail_mem=72.77 GB): 100%|██████████| 58/58 [00:01<00:00, 41.64it/s]


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
    Generated text:  Luca． I'm a Chinese boy． I have a good friend． Her name is Alice． She is very beautiful． She likes to eat delicious food． She likes to play with her toys． She is very friendly and helpful． When she's ill， she needs to see the doctor． She can't go to the doctor because she has to cook food． I can go to the doctor． Alice asks me to help her． But I don't know how to do it． I am very sorry． What should I do？ What should I do？ I don't know． （1）Where is Alice from？____ A．The
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide between two contracts.
    
    Contract A requires a 5% tax on all payments made to the president. The president has a total of $800,000 in annual payments. Contract B does not require any taxes, and the president has a total of $500,000 in annual payments. 
    
    The president wants to maximize the total revenue from these payments. Assuming the president wants to keep all of the money, how much should he spend on each contract to maximize the total revenue?
    To determine how much the president should spend on each contract to maximize the total revenue, we need to calculate the
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris. Paris is a city of contrasts, as well as a city of beauty. The city is often called the "city of light" and is a landmark in Europe. Paris is often called the "city of light" because it is a cosmopolitan city that attracts many people from all over the world. It is also known as the "city of light" because of its famous landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre. The city is also known for its rich culture and history. It is a major center of education and has many important universities. Paris is also known
    ===============================
    Prompt: The future of AI is
    Generated text:  here but it’s not yet ready for maturity
    
    How will you incorporate machine learning into your organization? Does your organization have the tools to use machine learning? Are you aware of the algorithms that exist and how they are implemented? These are all questions that you need to answer if you want to succeed in the future of AI.
    
    In a recent webinar, Maarten de Koning, partner at Kroll, and Patrick Sun, partner at the Boston Consulting Group (BCG), presented the future of AI and discussed the skills that are required to succeed in the field.
    
    The webinar was presented at the Blue Planet conference and took place on the 


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I enjoy [job title] because [reason for interest]. I'm always looking for ways to [job title] and I'm always eager to learn new things. What's your favorite hobby or activity? I love [hobby or activity]. I'm always looking for new ways to [hobby or activity] and I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for business, finance, and tourism. The city is known for its annual Eiffel Tower Festival and its annual fashion week. Paris is a popular tourist destination and a cultural hub for Europe. It is a major economic and political center in France and plays a significant role in the country's history and culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, AI is likely to continue to be used for ethical and social purposes, such as improving access to healthcare and reducing poverty. However, there are also potential risks and challenges associated with AI, such as job displacement and privacy concerns. As these issues are addressed, it is likely that AI will continue to play an increasingly important role in shaping our future.
    


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
    Generated text:  [Your Name] and I am [Age] years old. I have been a fan of storytelling since childhood, always eager to learn new techniques and skills. I am a versatile writer, able to produce creative, engaging, and original content. My writing is known for being both entertaining and thought-provoking, and I love sharing my stories with others. I am a passionate advocate for storytelling and hope to inspire others to embrace the power of words. What inspired you to become a writer? What are your goals for the future in the world of writing? My writing journey began in my childhood, when I was always drawn to the imagination
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is located in the south of the country and is known as the "City of Light" for its stunning architecture and rich cultural heritage. 
    
    (Note: The statement is true, but it doesn't address the question about the information provided in the response.) The capital of France is Paris, located in the south of the country and known as the "City of Light" for its stunning architecture and rich cultural heritage. 
    
    (Note: The statement is true, but it doesn't address the question about the information provided in the response.) 
    
    The capital of France is Paris, located in the south of the country and known as the "
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a variety of different trends and developments. Here are some of the most likely trends that could be present in the future:
    
    1. Personalized AI: With the increasing amount of data being generated by machines and humans alike, it is likely that AI will become more personalized. As AI is able to learn from data, it will become better at understanding and predicting human behavior, and will be able to tailor its responses to the specific needs and preferences of each individual.
    
    2. Autonomous AI: With the development of more advanced AI systems, it is likely that we will see the development of autonomous AI systems. These systems will be able


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

    ],

     and

     I

    'm

     [

    insert

     occupation

    ]

     by

     profession

    .

     I

    'm

     a

     kind

     and

     compassionate

     person

     who

     loves

     spending

     my

     days

     helping

     others

     in

     need

    .

     In

     my

     spare

     time

    ,

     I

     enjoy

     reading

    ,

     playing

     sports

    ,

     and

     volunteering

     at

     local

     animal

     shelters

    .
    


    [

    Insert

     about

     

    1

    0

    0

     words

     about

     yourself

    ,

     such

     as

    :

     "

    I

     have

     a

     deep

     empathy

     for

     those

     in

     need

    ,

     and

     I

     strive

     to

     make

     a

     positive

     impact

     in

     my

     community

    ."

    ]
    


    Good

    ,

     so

     that

    's

     the

     kind

     of

     character

     you

    're

     looking

     for

    .

     Do

     you

     have

     a

     particular

     character

     or

     movie

     or

     book

     in

     mind

     where

     you

     want

     to

     be

     featured

    ?

     I

    'm

     not

     sure

     which

     I

    'd

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     as

     the

     City

     of

     Light

    .

     
    


    Explanation

     for

     the

     above

     statement

    :

     
    


    The

     statement

     clearly

     indicates

     the

     name

     of

     the

     capital

     city

     of

     France

    ,

     referring

     to

     Paris

    ,

     and

     mentions

     that

     it

     is

     known

     as

     the

     City

     of

     Light

    .

     This

     information

     is

     crucial

     as

     it

     allows

     one

     to

     recognize

     the

     city

    's

     cultural

     significance

     and

     identify

     it

     as

     a

     major

     European

     capital

    .

     The

     name

     "

    City

     of

     Light

    "

     reflects

     the

     city

    's

     prominence

     in

     culture

    ,

     art

    ,

     and

     urban

     design

     during

     the

     

    1

    9

    th

     century

    .

     Therefore

    ,

     by

     providing

     this

     information

    ,

     one

     can

     gain

     a

     better

     understanding

     of

     the

     French

     capital

     and

     its

     role

     in

     European

     history

     and

     architecture

    .

     
    


    Furthermore

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     with

     a

     variety

     of

     potential

     trends

     and

     technologies

     shaping

     the

     direction

     of

     the

     field

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

     ethics

     and

     safety

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     lives

    ,

     there

     will

     be

     increasing

     pressure

     to

     address

     the

     ethical

     and

     safety

     concerns

     associated

     with

     its

     use

    .

     This

     includes

     issues

     such

     as

     bias

    ,

     accountability

    ,

     and

     the

     potential

     for

     AI

     to

     override

     human

     decision

    -making

    .
    


    2

    .

     More

     advanced

     models

    :

     As

     AI

     models

     become

     more

     complex

     and

     powerful

    ,

     we

     may

     see

     the

     development

     of

     new

     models

     that

     can

     perform

     tasks

     that

     are

     currently

     beyond

     the

     reach

     of

     current

     AI

     systems

    .

     For

     example

    ,

     deep

     learning

     models

     may

     become

     more

     adept

    



```python
llm.shutdown()
```
