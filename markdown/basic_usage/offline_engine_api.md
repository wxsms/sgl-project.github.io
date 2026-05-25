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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.51it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.95it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.95it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.74it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]

    Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 27.88it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 27.88it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 38.14it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 38.14it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 38.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.82it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.68it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.68it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.68it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.85it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.85it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.85it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.85it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.85it/s]

    Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:01, 29.97it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:01, 29.97it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  47%|████▋     | 27/58 [00:00<00:01, 29.97it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:01, 29.97it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.97it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.33it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.33it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.33it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.33it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.33it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.33it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.56it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.56it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.56it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.56it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.56it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.56it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=160 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 34.83it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.47it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.47it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.47it/s]Capturing num tokens (num_tokens=64 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.47it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.47it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.35it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.30it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.30it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.30it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.30it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 34.47it/s]


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
    Generated text:  Diana and I am the founder of the ECD model. I have been working for the past 10 years as a doctor in the mental health field with the mission of supporting patients to make decisions about their own health. As a pediatrician, I also regularly examine patients in the ED to see if they have symptoms of a mental health condition. I have a passion for helping people navigate the complex and often confusing world of mental health. I have a unique perspective on the ED, as the part of the healthcare system that treats both children and adults, and I understand the unique needs of patients in that part of the system. My team
    ===============================
    Prompt: The president of the United States is
    Generated text:  from which country?
    The president of the United States is directly elected by the citizens of the United States. Therefore, the country from which the president is from is the United States. It is important to note that while the president is directly elected by the citizens, he does not represent any particular region or state. He represents the entire United States. The United States is one of the world's largest and most populous countries, and it has a strong and diverse economy, a diverse population, and a rich history. Its leaders, the president and vice president, are often involved in national policy and governance. The country is divided into 50
    ===============================
    Prompt: The capital of France is
    Generated text:  _____.
    A. London
    B. Paris
    C. Berlin
    D. Moscow
    Answer: B
    
    The common thread that binds all activities within a product is
    A. Function
    B. Core Value
    C. Customer
    D. Resource
    Answer: A
    
    How many of the following statements are true?
    ① All countries have the right to withdraw from the international monetary system
    ② All countries have the right to participate in the international monetary system
    ③ The international monetary system is a social system
    ④ The international monetary system is a natural system
    A. 1 statement
    B. 2 statements
    
    ===============================
    Prompt: The future of AI is
    Generated text:  bright with the promise of making data more accessible, better-informed and more ethical. And the current generation of AI systems are a crucial part of that future.
    But to do that, AI systems need to learn from data and learn without bias. Current AI systems are prone to bias because they are trained on large amounts of data that comes from diverse sources and which are often biased against a particular group. By improving the data that the AI system uses to learn from, we can make sure that the AI system learns from data that is not biased. This means that we can create better, more ethical AI systems that can make data more accessible,


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


    Generated text:  [Name] and I am a [occupation] with [number of years] years of experience in [field]. I am a [character trait] and [character trait] that I am passionate about. I am [character trait] and [character trait] that I am passionate about. I am [character trait] and [character trait] that I am passionate about. I am [character trait] and [character trait] that I am passionate about. I am [character trait] and [character trait] that I am passionate about. I am [character trait] and [character trait] that I am passionate about. I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and is famous for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant culture, delicious cuisine, and annual festivals such as the Eiffel Tower Festival and the Louvre Festival. The city is home to many world-renowned museums, art galleries, and theaters, and is a popular tourist destination for visitors from around the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. Its status as the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation: One of the most significant trends in AI is the increasing automation of tasks that are currently performed by humans. This could include tasks such as data analysis, decision-making, and problem-solving, as well as tasks that are currently performed by machines.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be a growing need for enhanced privacy and security measures. This could include measures such as
    


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
    Generated text:  [Name], and I'm a [Job Title] at [Company]. My background is in [Field of Study], and I've had a passion for [Project/Goal] since [Year of Birth]. I've always been a [Personality Trait], and I'm passionate about [Attraction/Commitment]. I enjoy [Activity/Interest] and believe in [Leadership Philosophy]. I'm also an [Activity/Interest] enthusiast, and I've always been excited to learn and expand my knowledge. In my spare time, I enjoy [Activity/Interest] with friends and family. I'm always eager to learn and make
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is located in the north-central region of the country, near the Atlantic Ocean. It was founded in the 12th century and is the largest and most populous city in the European Union. Paris is known for its rich history, art, culture, and cuisine. It is home to the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cafes, theaters, and iconic landmarks such as the Champs-Élysées and the Eiffel Tower. Paris is a major cultural hub with a diverse population that includes many ethnic minorities and various religious groups.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and constantly evolving, with many potential trends emerging. Here are some potential areas that AI is likely to grow:
    
    1. Increased Integration with Human Capabilities: AI is becoming more integrated with human capabilities in areas such as healthcare, transportation, and education. AI is learning to better understand and interpret human emotions and behaviors, which could lead to more empathetic AI systems.
    
    2. Personalized AI: As AI systems become more sophisticated, they will become more personalized and able to learn and adapt to individual users or contexts. This could lead to more customized AI solutions that are better suited to specific needs.
    
    3. Autonomous and Ethical AI:


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

    Your

     occupation

    ]

     who

     has

     always

     been

     fascinated

     by

     [

    Your

     passion

     or

     hobby

    ].

     I

     enjoy

     [

    How

     you

     find

     your

     passion

     or

     hobby

    ],

     learning

     new

     things

    ,

     and

     exploring

     different

     parts

     of

     the

     world

    .

     I

    'm

     always

     looking

     for

     opportunities

     to

     share

     my

     knowledge

     and

     experience

     with

     others

    ,

     and

     I

    'm

     looking

     forward

     to

     discovering

     new

     things

     and

     making

     new

     connections

     with

     people

    .

     How

     can

     I

     apply

     my

     passion

     for

     [

    Your

     hobby

    ]

     to

     my

     work

    ?

     Let

     me

     know

     if

     you

     need

     any

     other

     information

     or

     if

     there

    's

     anything

     else

     I

     can

     help

     you

     with

    .

     [

    Name

    ]

     [

    Occup

    ation

    ]


    [

    Your

     name

    ]

     [

    Your

     occupation

    ]


    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     population

     and

     the

     seat

     of

     the

     French

     government

    .

     It

     is

     home

     to

     many

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     a

     cultural

     hub

    ,

     hosting

     numerous

     events

     and

     festivals

     throughout

     the

     year

    .

     Paris

     is

     also

     a

     world

    -ren

    owned

     fashion

     and

     music

     capital

    ,

     with

     many

     famous

     fashion

     and

     music

     institutions

     and

     venues

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     including

     medieval

     ruins

    ,

     Renaissance

     art

    ,

     and

     modern

     architecture

    .

     It

     is

     a

     vibrant

     and

     dynamic

     urban

     center

    ,

     and

     Paris

     remains

     a

     major

     cultural

     center

     in

     Europe

    .

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     combination

     of

     high

     levels

     of

     automation

    ,

     self

    -learning

     and

     decision

    -making

    ,

     and

     the

     ability

     to

     interact

     with

     humans

     in

     a

     more

     natural

     and

     effective

     manner

    .

     Here

     are

     some

     of

     the

     most

     likely

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

     The

     use

     of

     AI

     and

     machine

     learning

     in

     manufacturing

    ,

     agriculture

    ,

     transportation

    ,

     and

     other

     industries

     is

     likely

     to

     continue

     growing

    .

     Automation

     can

     improve

     efficiency

     and

     reduce

     costs

    ,

     while

     also

     freeing

     up

     employees

     to

     focus

     on

     more

     complex

     tasks

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     likely

     to

     become

     more

     common

    ,

     with

     the

     ability

     to

     make

     decisions

     and

     take

     control

     of

     the

     vehicle

     based

     on

     the

     environment

     and

     other

     sensors

    .

     This

     could

    



```python
llm.shutdown()
```
