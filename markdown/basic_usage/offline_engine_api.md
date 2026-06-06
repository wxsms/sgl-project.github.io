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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.31it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.33it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.71it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.71it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.71it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.71it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.71it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.71it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.71it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.71it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.71it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.71it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.61it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.61it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.61it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.61it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.61it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.61it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.61it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.61it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.61it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.61it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.61it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.56it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.56it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.56it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.56it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.56it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.56it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.56it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.56it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.56it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.56it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.37it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:03, 14.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:03, 14.09it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 14.09it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   7%|▋         | 4/58 [00:00<00:03, 15.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   7%|▋         | 4/58 [00:00<00:03, 15.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.77 GB):   7%|▋         | 4/58 [00:00<00:03, 15.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.77 GB):  10%|█         | 6/58 [00:00<00:03, 16.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.76 GB):  10%|█         | 6/58 [00:00<00:03, 16.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):  10%|█         | 6/58 [00:00<00:03, 16.37it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.26it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.26it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.26it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.12it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.99it/s]Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.99it/s] Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.99it/s]Capturing num tokens (num_tokens=832 avail_mem=71.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.99it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.99it/s]

    Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.99it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.48it/s]Capturing num tokens (num_tokens=640 avail_mem=71.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.48it/s]Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.48it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.48it/s]Capturing num tokens (num_tokens=480 avail_mem=71.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.48it/s]Capturing num tokens (num_tokens=448 avail_mem=71.70 GB):  45%|████▍     | 26/58 [00:01<00:00, 38.48it/s]Capturing num tokens (num_tokens=448 avail_mem=71.70 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=352 avail_mem=71.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]

    Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=288 avail_mem=71.68 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=288 avail_mem=71.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.26it/s]Capturing num tokens (num_tokens=256 avail_mem=71.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.26it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.26it/s]Capturing num tokens (num_tokens=224 avail_mem=71.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.26it/s]Capturing num tokens (num_tokens=208 avail_mem=71.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.26it/s]Capturing num tokens (num_tokens=192 avail_mem=71.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.26it/s]Capturing num tokens (num_tokens=192 avail_mem=71.67 GB):  71%|███████   | 41/58 [00:01<00:00, 43.43it/s]Capturing num tokens (num_tokens=176 avail_mem=71.66 GB):  71%|███████   | 41/58 [00:01<00:00, 43.43it/s]Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  71%|███████   | 41/58 [00:01<00:00, 43.43it/s]Capturing num tokens (num_tokens=144 avail_mem=71.66 GB):  71%|███████   | 41/58 [00:01<00:00, 43.43it/s]

    Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  71%|███████   | 41/58 [00:01<00:00, 43.43it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  71%|███████   | 41/58 [00:01<00:00, 43.43it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.32it/s]Capturing num tokens (num_tokens=96 avail_mem=71.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.32it/s] Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.32it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.32it/s]Capturing num tokens (num_tokens=48 avail_mem=71.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.32it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.32it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=28 avail_mem=71.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=24 avail_mem=71.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.85it/s]

    Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=12 avail_mem=71.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.85it/s]Capturing num tokens (num_tokens=12 avail_mem=71.62 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.62it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.62it/s] Capturing num tokens (num_tokens=4 avail_mem=71.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.62it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 35.80it/s]


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
    Generated text:  Sarah and I am a product manager with a passion for finding new ways to improve the lives of people by creating something new. I have a Bachelor's degree in Computer Science and am currently pursuing a Master's degree in Product Management at Harvard Business School. I am thrilled to be a part of the Adam Group team and I look forward to bringing my expertise to the team. How can I become a part of the Adam Group team?
    As an AI language model, I cannot discuss political topics. However, I can provide you with some general information about the Adam Group.
    The Adam Group is a world-renowned organization that provides innovative solutions to complex
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to finalize a new executive order.
    
    The president has a list of 20 states, and the number of days each state has to sign a pledge to be a founding member of the United States. If the president has 100 days to review the pledges, and each state takes 3 days to sign, how many days will each state still have to sign before the order is signed? To determine how many days each state still has to sign the pledge before the order is signed, we can follow these steps:
    
    1. Identify the total number of days available for review.
    2. Determine the number of days each state has
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. Brussels
    C. Nice
    D. Strasbourg
    Answer:
    
    A
    
    Patient, male, 46 years old, admitted to the hospital due to acute myocardial infarction. The patient is alert, with a blood pressure of 140/90mmHg, clear consciousness, and a heart rate of 80 beats per minute. The appropriate position for this patient is
    A. Supine position
    B. Head low foot high position
    C. Lateral position
    D. Right lateral position
    E. Left lateral position
    Answer:
    
    E
    
    Which of the
    ===============================
    Prompt: The future of AI is
    Generated text:  fast moving, and the rapid changes in technology and industry will require rapid and efficient development of new AI algorithms. However, training data for AI models, which is critical for their performance, is often an outdated and inefficient source of data. In this article, we explore the benefits of using data collection and analysis methods in the development of AI models. We discuss the importance of using real data in AI, the value of using real data in the development of AI models, and the importance of collecting and analyzing real data for the development of AI models. We also discuss the challenges associated with using real data and provide recommendations for overcoming these challenges. Finally


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [Type of Person] who is [What you do for a living]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do for a living? I enjoy [What you do for a living]. I'm always looking for new ways to improve myself and make a positive impact on the world. What do you like to do for a living? I enjoy [What you do for a living]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is the capital of France and the largest city in the European Union. The city is known for its fashion industry, art scene, and cuisine. It is also home to the French Parliament, the French National Museum of Modern Art, and the Eiffel Tower. Paris is a city of contrasts, with its modern architecture and historical landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations. This could lead to more complex and sophisticated AI systems that can perform tasks that are currently beyond the capabilities of human intelligence.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, such as privacy, fairness, and accountability. This will require developers
    


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
    Generated text:  Jane, and I'm a self-employed marketing manager for a small software development company. I'm passionate about helping businesses grow and achieve their goals through effective marketing strategies. My goal is to provide my clients with a holistic approach to their marketing efforts, covering everything from market research to campaign execution. I'm excited about the opportunity to work with a talented team of creative and hard-working individuals, and I'm constantly looking for new challenges and opportunities to grow and learn. I'm patient, reliable, and always willing to adapt to new ideas and approaches. Thank you for considering my application for this position. As an AI language model, I can generate
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its iconic architecture, vibrant culture, and international status.
    
    The capital of France is Paris, known for its iconic architecture, vibrant culture, and international status. The city's rich history, including its role as a seat of power and influence for many centuries, has shaped its distinctive character. Paris is also home to some of the world's most famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city's significance has made it a popular tourist destination, attracting millions of visitors each year. Paris is a unique and captivating blend of modernity and tradition, making it
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and it is expected to continue to develop rapidly. Some possible future trends in AI include:
    
    1. Increased integration with human AI: AI is expected to become even more integrated with humans, enabling them to collaborate, make decisions, and even make ethical decisions.
    
    2. Developing new types of AI: AI is also expected to be developing new types of AI, such as deep learning, natural language processing, and computer vision, that are more powerful and capable than what we have today.
    
    3. AI becoming more accessible: AI is becoming more accessible to more people, as AI algorithms are getting better and more accurate. This means that AI


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

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ]

     person

    .

     I

    'm

     a

     [

    Occup

    ation

    ]

     with

     a

     passion

     for

     [

    Career

     objective

     or

     hobby

    ].

     I

     enjoy

     [

    What

     I

     like

     to

     do

     for

     fun

    ,

     including

     hobbies

    ,

     sports

    ,

     movies

    ,

     music

    ,

     etc

    .

    ].

     I

    've

     always

     been

     [

    An

     accomplishment

     or

     achievement

     that

     gives

     me

     pride

    ,

     like

     winning

     a

     contest

    ,

     volunteering

    ,

     etc

    .

    ].

     I

    'm

     always

     looking

     for

     [

    What

     I

    'm

     looking

     for

     in

     a

     partner

     or

     a

     friend

    ,

     such

     as

     personal

     qualities

    ,

     interests

    ,

     etc

    .

    ].

     I

    'm

     passionate

     about

     [

    What

     I

    'm

     passionate

     about

    ,

     such

     as

     nature

    ,

     technology

    ,

     history

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     sprawling

     city

     with

     a

     rich

     history

     and

     culture

    .

     It

     is

     located

     in

     the

     north

     of

     the

     country

     and

     is

     the

     largest

     city

     in

     Europe

    .

     Paris

     has

     been

     a

     major

     center

     for

     commerce

    ,

     industry

    ,

     and

     culture

     since

     ancient

     times

     and

     is

     known

     for

     its

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

     It

     is

     also

     a

     vibrant

     center

     for

     art

    ,

     music

    ,

     and

     fashion

    .

     Despite

     its

     size

    ,

     Paris

     is

     a

     relatively

     safe

     city

     for

     travelers

    .

     It

     is

     home

     to

     many

     different

     communities

    ,

     with

     diverse

     cultural

     practices

    ,

     traditions

    ,

     and

     food

    .

     The

     French

     government

     plays

     a

     significant

     role

     in

     shaping

     Paris

    's

     culture

     and

     economy

    ,

     while

     also

     working

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

    ,

     and

     there

     are

     several

     trends

     that

     are

     likely

     to

     shape

     the

     landscape

     of

     this

     technology

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     can

     be

     used

     to

     improve

     the

     accuracy

     and

     efficiency

     of

     medical

     diagnoses

    ,

     predict

     patient

     outcomes

    ,

     and

     develop

     personalized

     treatment

     plans

    .

     AI

    -powered

     imaging

     systems

     can

     also

     help

     doctors

     identify

     early

     signs

     of

     disease

    .
    


    2

    .

     AI

     in

     education

    :

     AI

     is

     being

     used

     to

     develop

     personalized

     learning

     paths

    ,

     adaptive

     learning

     systems

    ,

     and

     tutoring

     assistants

    .

     These

     tools

     can

     help

     students

     learn

     at

     their

     own

     pace

     and

     in

     their

     own

     way

    ,

     which

     could

     improve

     their

     learning

     outcomes

    .
    


    3

    .

     AI

     in

     manufacturing

    :

    



```python
llm.shutdown()
```
