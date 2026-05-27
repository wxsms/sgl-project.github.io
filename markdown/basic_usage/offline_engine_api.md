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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:15,  3.20it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:15,  3.20it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:15,  3.20it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:15,  3.20it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:15,  3.20it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:15,  3.20it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:15,  3.20it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:15,  3.20it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:04<00:15,  3.20it/s]Compiling num tokens (num_tokens=1792):  16%|█▌        | 9/58 [00:04<00:15,  3.20it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:04,  8.33it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:04,  8.33it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:04,  8.33it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:04,  8.33it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:04,  8.33it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:04<00:04,  8.33it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:04<00:04,  8.33it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:04<00:04,  8.33it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:04<00:04,  8.33it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:02, 13.90it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:02, 13.90it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:04<00:02, 13.90it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:02, 13.90it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:04<00:02, 13.90it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:04<00:02, 13.90it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:04<00:02, 13.90it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:04<00:02, 13.90it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:04<00:02, 13.90it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:04<00:02, 13.90it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:01, 21.38it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:01, 21.38it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:01, 21.38it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:01, 21.38it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:04<00:01, 21.38it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:04<00:01, 21.38it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:04<00:01, 21.38it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:04<00:01, 21.38it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:04<00:01, 21.38it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:04<00:01, 21.38it/s]

    Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:04<00:00, 29.87it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:04<00:00, 29.87it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:04<00:00, 29.87it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:04<00:00, 29.87it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:04<00:00, 29.87it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:04<00:00, 29.87it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:04<00:00, 29.87it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:04<00:00, 29.87it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:04<00:00, 29.87it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:04<00:00, 29.87it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 38.80it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 38.80it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 38.80it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 38.80it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 38.80it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 38.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.21it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.21it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.21it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.21it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.21it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.04it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.04it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.04it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.04it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.04it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.04it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.14it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.81it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.81it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.81it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.81it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.81it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.81it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.56it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.56it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.56it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.56it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.56it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.67it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.67it/s] Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  81%|████████  | 47/58 [00:01<00:00, 30.95it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 30.95it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 30.95it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 30.95it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 30.95it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 29.88it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 29.88it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 29.88it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 32.47it/s]


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
    Generated text:  Xiaowu and I'm a 16 year old, and I'm planning to do a Career Exploration. Please help me with some questions to help me find the right career for me.
    
    Certainly! I'd be happy to help you with your career exploration. Here are some general questions to guide you in identifying the right career for you:
    
    1. **Understanding Your Interests and Skills:**
       - What are your strengths and areas of interest? Are you passionate about something?
       - What skills are you already confident in, or do you see yourself developing these in the future?
    
    2. **Career Goals:**
       - What
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she helps to run the country. This person is also called the president of the United States. This person is the leader of the country. He or she helps to make laws and take care of the country. What kind of jobs does the president have? The president is in charge of the United States government. He or she works with other important people to make important laws. The president has many important jobs. He or she helps the country to make and enforce laws. They also help to make the president and other important people called cabinet members (the cabinet) work well with each other. The president has
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. London
    C. Moscow
    D. New York
    Answer:
    A
    
    The ancient city of ___ is the largest city in the UK, known as the 'City of London'.
    A. Cambridge
    B. London
    C. Edinburgh
    D. Bristol
    Answer:
    B
    
    According to the latest estimate, by the end of 2022, the total number of people in the UK was approximately ____ million.
    A. 600
    B. 700
    C. 800
    D. 900
    Answer:
    B
    
    When a baby
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but there are a few things we know for sure. In the future, AI will play a critical role in shaping our world, whether it be in areas such as healthcare, transportation, education, or financial services. Here are some of the key areas where AI will have the most impact:
    
    1. Healthcare: AI will revolutionize healthcare by providing doctors with more accurate diagnoses, predicting the risk of disease, and helping to personalize treatment plans. AI can also help in the development of new medications and therapies.
    
    2. Transportation: AI is already being used in the transportation industry to improve efficiency and reduce costs. It can be used for


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I've been with the company for [number of years] years. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] who is always looking for ways to [job title] my skills and knowledge. I'm a [job title] who is always looking for ways to [job title] my skills and knowledge. I'm a [job title] who is always looking for ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, as well as its rich cultural heritage and diverse cuisine. The city is also home to many world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination and a major economic center in France. It is also known for its fashion industry and its role in the French Revolution and the French Revolution. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced privacy and security: As AI systems become more integrated with human intelligence, there will be an increased need for privacy and security measures to protect the data and personal information
    


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
    Generated text:  [Name] and I'm a [Job Title] at [Company Name], where I'm currently [position]. I have a passion for [career objective] and I'm looking forward to [goals for the next project or project]. As a [personality trait or quality] [Name], I always try to [positive trait or quality that relates to my character] and I'm always [positive trait or quality that relates to my character]. My goal is to [goals for the next project or project], and I'm constantly learning and growing as a result. I'm a [personality trait or quality] and I strive to [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the most populous city in Europe and is the heart of the nation. The city is known for its rich cultural heritage, iconic landmarks such as the Eiffel Tower, and its role in European affairs. Paris is also home to the Eiffel Tower and Montmartre, where renowned writers and artists have lived and worked. As a cosmopolitan city, Paris is home to a diverse range of people, from visitors and tourists to locals and residents alike. The city is known for its delicious cuisine, stunning architecture, and lively nightlife, making it a popular tourist destination. Despite its size and prominence, Paris is still
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and has the potential to revolutionize industries, advance scientific knowledge, and improve human life. Here are some possible trends in the future of AI:
    
    1. Increased autonomy and decision-making: As AI becomes more capable of making decisions based on data, it will become even more autonomous and capable of making decisions without human intervention.
    
    2. Greater personalization and tailored experiences: With the ability to analyze and understand user data, AI will be able to provide more tailored and personalized experiences for individuals.
    
    3. Enhanced security and privacy: As AI is used in various sectors, there will be more opportunities for cybersecurity and privacy issues.
    
    4. Autonomous vehicles


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

     __

    ________

     and

     I

     am

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    ,

     and

     I

     have

     __

    ________

    .

     I

     enjoy

     __

    ________

     and

     __

    ________

    ,

     and

     I

     always

     try

     to

     __

    ________

    .

     I

    'm

     __

    ________

    ,

     and

     I

     like

     to

     __

    ________

    .

     I

    'm __

    ________

    ,

     and

     I

     have

     __

    ________

    .
    


    [

    Add

     any

     other

     relevant

     information

     about

     the

     character

    ]

     [

    Add

     any

     other

     relevant

     information

     about

     the

     character

    ]
    


    I

     hope

     you

     enjoy

     this

     short

    ,

     neutral

     self

    -int

    roduction

    .

     Let

     me

     know

     if

     you

     would

     like

     me

     to

     add

     any

     additional

     details

     or

     modify

     any

     information

    .

     Good

     luck

     with

     your

     self

    -int

    roduction

    !

     

    📞

    💬

    👨

    ‍

    👩

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     significant

     cultural

     and

     economic

     center

    .

     The

     city

     was

     founded

     in

     

    7

    8

    9

     AD

     and

     is

     known

     for

     its

     vibrant

     arts

     scene

    ,

     rich

     history

    ,

     and

     high

     standard

     of

     living

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     the

     country

    's

     cultural

     and

     political

     capital

    .

     Paris

     is

     a

     major

     hub

     for

     science

    ,

     technology

    ,

     and

     innovation

     and

     has

     a

     thriving

     food

     industry

    .

     It

     is

     also

     home

     to

     numerous

     cultural

     institutions

     and

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     Paris

     is

     a

     major

     hub

     for

     business

     and

     finance

    ,

     and the

     French

     Parliament

     is

     located

     in

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     Despite

     its

     size

    ,

     Paris

     offers

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     challenges

    .

     Here

     are

     some

     possible

     trends

     in

     the

     field

    :
    


    1

    .

     Increased

     AI

     ethics

     and

     transparency

    :

     As

     AI

     systems

     become

     more

     complex

     and

     advanced

    ,

     there

     will

     be

     increasing

     pressure

     to

     ensure

     that

     AI

     systems

     are

     developed

     and

     deployed

     in

     a

     way

     that

     is

     ethical and

     transparent

    .
    


    2

    .

     Development

     of

     more

     advanced

     AI

     technologies

    :

     AI

     is

     expected

     to

     continue

     to

     advance

     at

     an

     unprecedented

     rate

     in

     the

     coming

     years

    ,

     with

     new

     breakthrough

    s

     in

     areas

     such

     as

     deep

     learning

    ,

     natural

     language

     processing

    ,

     and

     computer

     vision

    .
    


    3

    .

     AI

     integration

     with

     human

     behavior

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     there

     will

     be

     a

     growing

     focus

     on

     how

     these

     systems

     can

     be

     integrated

    



```python
llm.shutdown()
```
