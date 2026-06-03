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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.08it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.08it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.08it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.08it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.17it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.17it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.17it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.17it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.17it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.17it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.17it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.17it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.17it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.15it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.15it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.15it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.15it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.15it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.15it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.15it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.15it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.15it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.15it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.15it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 21.88it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 21.88it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 21.88it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 21.88it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 21.88it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 21.88it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 21.88it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 21.88it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 21.88it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 21.88it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 21.88it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 30.71it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.40 GB):   3%|▎         | 2/58 [00:00<00:03, 17.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.39 GB):   3%|▎         | 2/58 [00:00<00:03, 17.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.39 GB):   3%|▎         | 2/58 [00:00<00:03, 17.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.39 GB):   3%|▎         | 2/58 [00:00<00:03, 17.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.37 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.37 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.37 GB):   9%|▊         | 5/58 [00:00<00:02, 21.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=61.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.56it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.34 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=61.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s]Capturing num tokens (num_tokens=960 avail_mem=61.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.39it/s] Capturing num tokens (num_tokens=960 avail_mem=61.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=896 avail_mem=61.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=832 avail_mem=61.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=768 avail_mem=61.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=704 avail_mem=61.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=640 avail_mem=61.31 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=640 avail_mem=61.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=576 avail_mem=61.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=512 avail_mem=61.30 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=480 avail_mem=61.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.58it/s]

    Capturing num tokens (num_tokens=448 avail_mem=61.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=416 avail_mem=61.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=416 avail_mem=61.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=384 avail_mem=61.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=352 avail_mem=61.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=320 avail_mem=61.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=288 avail_mem=61.29 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=256 avail_mem=61.29 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.26it/s]Capturing num tokens (num_tokens=256 avail_mem=61.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=240 avail_mem=61.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=224 avail_mem=61.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=208 avail_mem=61.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.53it/s]

    Capturing num tokens (num_tokens=192 avail_mem=61.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=176 avail_mem=61.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=176 avail_mem=61.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=160 avail_mem=61.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=144 avail_mem=61.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=128 avail_mem=61.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=112 avail_mem=61.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=96 avail_mem=61.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.53it/s] Capturing num tokens (num_tokens=96 avail_mem=61.26 GB):  81%|████████  | 47/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=80 avail_mem=61.26 GB):  81%|████████  | 47/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=64 avail_mem=61.25 GB):  81%|████████  | 47/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=48 avail_mem=61.25 GB):  81%|████████  | 47/58 [00:01<00:00, 44.92it/s]

    Capturing num tokens (num_tokens=32 avail_mem=61.25 GB):  81%|████████  | 47/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=28 avail_mem=61.24 GB):  81%|████████  | 47/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=28 avail_mem=61.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.04it/s]Capturing num tokens (num_tokens=24 avail_mem=61.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.04it/s]Capturing num tokens (num_tokens=20 avail_mem=61.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.04it/s]Capturing num tokens (num_tokens=16 avail_mem=61.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.04it/s]Capturing num tokens (num_tokens=12 avail_mem=61.23 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.04it/s]Capturing num tokens (num_tokens=8 avail_mem=61.23 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.04it/s] Capturing num tokens (num_tokens=8 avail_mem=61.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=4 avail_mem=61.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=4 avail_mem=61.23 GB): 100%|██████████| 58/58 [00:01<00:00, 39.73it/s]


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
    Generated text:  Sara, I am a 17 year old high school senior and I recently had a really fun day. I had a really good time playing with my friends in the neighborhood. I had a great time talking to my family, and the best part of the day was that I got to relax and have fun with my friends and family. I had a great time. I really enjoyed it.
    Question: How would you describe the narrator's day? 
    Pick from:
    (i) very good
    (ii) just okay
    (iii) not at all
    (iv) very bad
    Answer: (i) very good
    The narrator's day
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. As of now, the president is Barack Obama. He was born on September 30, 1961, in Honolulu, Hawaii, United States. Obama was elected to the office of the United States President on January 20, 2009, and has been serving since then.
    
    Given the information provided, what is the president's role?
    Based on the information provided, the president's role is serving as the President of the United States. Specifically, Obama has been serving since 2009, as the country has been undergoing changes in the government leadership.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris was founded in 785 and is located in the center of the Mediterranean Sea on the left bank of the Seine. It is the world's most populous city, with a population of around 2.8 million people. It is the largest city in France by area. It's also the largest city in the world by population. Paris was first called Parisa. It was the name of the first church in the area of the Languedoc. The word Paris means "the city of the open fields". In 1204, Louis IX, King of France was crowned King of France.
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it is also risky. We see a number of potential dangers in our journey towards AI, with the biggest danger being that we create an artificial intelligence that is as powerful as a human being, but ultimately less intelligent.
    
    There are many reasons for this. One of them is that we do not have the tools to program the artificial intelligence to have the same cognitive abilities as a human being. This makes it difficult to create AI that can effectively reason, think, and solve problems like a human being can.
    
    Another reason for this is that we do not have enough data to train an AI to accurately replicate human-like abilities. If we


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [job title] at [company name], and I've been working in [industry] for [number of years] years. I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field. What excites you about your job? I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field. What do you like to do in your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a cultural and economic hub, with a diverse population and a rich history that continues to shape the city's identity. It is a popular tourist destination and a major center for business and finance. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI systems, there will be a greater emphasis on ethical considerations and the development of AI that is designed to be fair, transparent, and accountable.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, from self-driving cars to smart homes. As more companies and governments invest in AI, it is likely that we will see more integration
    


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
    Generated text:  [Name], and I am a [occupation] who have always been fascinated by the beauty of nature. I have a knack for capturing the essence of the world around me and making it come alive in the words I write. I am passionate about using my writing to help others connect with nature, and to inspire them to live a sustainable life. I am a storyteller, and I bring a unique perspective to any project I undertake. I enjoy the challenge of creating something beautiful and meaningful from the very ground of the earth. As a storyteller, I believe in the power of storytelling to inspire and connect people. I am excited to tell
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement is accurate, as it correctly identifies Paris as the official capital city of France, according to the official United Nations website. Paris is known for its historical and cultural landmarks, such as Notre-Dame Cathedral and the Eiffel Tower, and for its vibrant and diverse population. It is also one of the world's most important financial and business capitals. The city has been featured in numerous films and has a rich cultural heritage, making it a popular tourist destination for visitors from all over the world. Paris is also known for its cuisine and its contributions to art, literature, and music. Its stunning location in the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, with many exciting developments on the horizon. Here are some potential trends in AI that are likely to shape the future:
    
    1. Increased transparency and accountability: As AI systems become more complex and powerful, they will need to be more transparent and accountable. This means that we will see more data available to users, more explanations for decisions, and more methods for verifying AI systems.
    
    2. Enhanced empathy and emotional intelligence: AI systems will be able to learn and respond to the emotions of their users, leading to more empathetic and emotionally intelligent AI.
    
    3. Improved natural language processing: As AI systems become more capable of processing and understanding natural


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

    First

     Name

    ]

     and

     I

     am

     [

    Last

     Name

    ].

     I

    've

     always

     had

     a

     fascination

     with

     the

     science

     of

     things

    ,

     particularly

     the

     physical

     aspects

     of

     biology

     and

     medicine

    .

     I

    've

     always

     been

     fascinated

     by

     the

     natural

     world

     and

     the

     ways

     in

     which

     it

     operates

    .

     I

     enjoy

     learning

     about

     the

     wonders

     of

     the

     world

     around

     me

    ,

     and

     I

     try

     to

     approach

     my

     work

     with

     a

     curious

     and

     open

    -minded

     attitude

    .

     I

     love

     to

     connect

     with

     people

     and

     help

     others

     in

     my

     efforts

     to

     understand

     the

     world

     better

    .

     And

     that

    's

     why

     I

    'm

     here

     to

     share

     my

     knowledge

     and

     insights

     with

     you

    .

     What

    's

     your

     name

    ?

     And

     what

     is

     your

     profession

    ?

     Nice

     to

     meet

     you

    ,

     [

    First

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    ,

     which

     is

     a

     bustling

     met

    ropolis

     with

     a

     rich

     cultural

     heritage

    .

     It

     is

     home

     to

     numerous

     historic

     landmarks

     and

     is

     a

     popular

     tourist

     destination

    ,

     with

     its

     iconic

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

     standing

     as

     reminders

     of

     its

     French

     past

    .

     Paris

     is

     also

     a

     major

     economic

     and

     financial

     center

    ,

     with

     its

     central

     business

     district

     and

     shopping

     district

     attracting

     millions

     of

     visitors

     each

     year

    .

     Despite

     the

     city

    's

     size

    ,

     Paris

     remains

     a

     relatively

     compact

     urban

     area

    ,

     with

     a

     low

     population

     density

     and

     a

     focus

     on

     its

     cultural

     and

     historical

     significance

    .

     
    


    Paris

     is

     often

     referred

     to

     as

     a

     "

    City

     of

     Lights

    "

     due

     to

     its

     lighting

     up

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     technology

    ,

     changes

     in

     societal

     values

    ,

     and

     new

     ways

     of

     understanding

     and

     interacting

     with

     the

     world

     around

     us

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

     that

     could

     shape

     the

     future

     of

     the

     field

    :
    


    1

    .

     Increased

     sophistication

    :

     With

     continued

     advancements

     in

     AI

    ,

     we

     can

     expect

     to

     see

     greater

     sophistication

     in

     the

     technology

     used

     to

     train

     and

     deploy

     AI

     systems

    .

     This

     could

     lead

     to

     even

     more

     advanced

     models

     that

     can

     perform

     more

     complex

     tasks

     with

     greater

     accuracy

     and

     speed

    .
    


    2

    .

     Personal

    ization

    :

     AI

     is

     already

     being

     used

     to

     personalize

     the

     experience

     of

     customers

    ,

     which

     will

     likely

     become

     even

     more

     sophisticated

     in

     the

     future

    .

     By

    



```python
llm.shutdown()
```
