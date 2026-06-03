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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.69it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:55,  5.19s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:55,  5.19s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:55,  5.19s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:55,  5.19s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:55,  5.19s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.40it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.40it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.40it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.40it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.40it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.40it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.40it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.40it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.40it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.16it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.16it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.16it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.16it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.16it/s]

    Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.16it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.16it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.16it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 11.29it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 11.29it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 11.29it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 11.29it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 11.29it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 11.29it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 11.29it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 11.29it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 16.18it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 16.18it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 16.18it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 16.18it/s]

    Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 16.18it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 16.18it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 16.18it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 25.64it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 25.64it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 25.64it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 25.64it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 25.64it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:06<00:00, 25.64it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:06<00:00, 25.64it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 31.03it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 31.03it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 31.03it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 31.03it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 31.03it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 31.03it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 31.03it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 31.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.31 GB):   3%|▎         | 2/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.30 GB):   3%|▎         | 2/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.28 GB):   3%|▎         | 2/58 [00:00<00:03, 14.66it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.28 GB):   7%|▋         | 4/58 [00:00<00:03, 16.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.28 GB):   7%|▋         | 4/58 [00:00<00:03, 16.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.29 GB):   7%|▋         | 4/58 [00:00<00:03, 16.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.28 GB):   7%|▋         | 4/58 [00:00<00:03, 16.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.28 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.27 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.26 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.26 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.26 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.25 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.24 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.23 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.23 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.23 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.20it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.20 GB):  31%|███       | 18/58 [00:00<00:01, 28.56it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.20 GB):  31%|███       | 18/58 [00:00<00:01, 28.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.20 GB):  31%|███       | 18/58 [00:00<00:01, 28.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.17 GB):  31%|███       | 18/58 [00:00<00:01, 28.56it/s]Capturing num tokens (num_tokens=960 avail_mem=74.18 GB):  31%|███       | 18/58 [00:00<00:01, 28.56it/s] Capturing num tokens (num_tokens=960 avail_mem=74.18 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.92it/s]Capturing num tokens (num_tokens=896 avail_mem=74.16 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.92it/s]Capturing num tokens (num_tokens=832 avail_mem=74.17 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.92it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.16 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.92it/s]Capturing num tokens (num_tokens=704 avail_mem=74.16 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.92it/s]Capturing num tokens (num_tokens=704 avail_mem=74.16 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.67it/s]Capturing num tokens (num_tokens=640 avail_mem=74.15 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.67it/s]Capturing num tokens (num_tokens=576 avail_mem=74.15 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.67it/s]Capturing num tokens (num_tokens=512 avail_mem=74.13 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.67it/s]Capturing num tokens (num_tokens=480 avail_mem=74.14 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.67it/s]Capturing num tokens (num_tokens=480 avail_mem=74.14 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=448 avail_mem=74.14 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=416 avail_mem=74.13 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.31it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.12 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  52%|█████▏    | 30/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.73it/s]Capturing num tokens (num_tokens=320 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.73it/s]Capturing num tokens (num_tokens=288 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.73it/s]Capturing num tokens (num_tokens=256 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.73it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.73it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=208 avail_mem=74.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.62it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=144 avail_mem=74.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.43it/s] Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  81%|████████  | 47/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=80 avail_mem=74.08 GB):  81%|████████  | 47/58 [00:01<00:00, 38.64it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  81%|████████  | 47/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  81%|████████  | 47/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  81%|████████  | 47/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 35.64it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.29it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.29it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.29it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 32.28it/s]


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
    Generated text:  Larry.
    
    I'm from the Southern United States, and I have lived here for over 30 years. I have spent a great deal of my life working as a teacher, and I have spent a great deal of my life teaching English as a Second Language. 
    
    I taught English at the University of Texas at Austin for 13 years, and then left the university to pursue my academic career in physics, and I am currently teaching English at the University of Missouri. 
    
    I have been a life-long learner of English, and I have studied in the United Kingdom, and I have studied in the United States. I have studied at
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many armed soldiers to have on his country. He has two choices. In the first choice, he would like to have 30% of the total population as soldiers. In the second choice, he would like to have 20% of the total population as soldiers. However, each choice would result in a large price tag of $25 billion. If the price of military personnel increases by 10% each year, how many people would he have as soldiers in the long run if he chooses the first choice?
    
    To determine the number of soldiers the president of the United States would have if he chooses
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and it has a population of 2.6 million. What is the number of people in Paris if France has 33 million people in total?
    
    To determine the number of people in Paris, we start by noting that the total population of France is 33 million. The capital of France, Paris, has a population of 2.6 million. The population of Paris is the total population of France minus the population of the capital. Let's calculate this step-by-step.
    
    1. Identify the total population of France:
       \[
       \text{Total population of France} = 33,00
    ===============================
    Prompt: The future of AI is
    Generated text:  starting to look bleak, but it might not be the bleak one we think. In a new report, The Future of AI, published by the World Economic Forum, the authors are optimistic about the future of AI and the opportunities it will bring to the world.
    Although AI technologies are still new and will require further investment in research and development, the authors believe that AI has the potential to transform the world for the better. Here are some key points from the report:
    1. AI is expected to play a significant role in the areas of healthcare, transportation, and education.
    2. AI technologies will make it easier to automate mundane tasks and save


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many of the world's most famous museums and attractions. Paris is a vibrant and diverse city with a rich history and a strong sense of French identity. It is the largest city in France and the second-largest city in the European Union by population. The city is also home to many international organizations and institutions, including the French Academy of Sciences and the French Academy of Fine Arts. Paris is a popular tourist destination and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread adoption in healthcare, particularly in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection, risk management, and investment decision-making. As AI
    


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
    Generated text:  Jane, and I'm a professional copywriter with over 15 years of experience in crafting compelling copy for media like TV, social media, and print. I have a keen eye for detail, a strong sense of humor, and a natural talent for persuasive storytelling. I'm always up for a challenge, eager to learn and grow, and driven by a passion for creating impactful content that resonates with readers. Whether it's writing copy for a podcast or working on a social media campaign, I'm constantly pushing myself to improve my craft and add value to my clients' projects. I'm confident in my ability to craft stories that speak
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is located in the south of the country and is known for its rich history, art, and cuisine. Paris is also a major cultural and economic hub, with numerous museums, art galleries, and other cultural institutions. The city has a diverse population of approximately 2.1 million people, making it one of the largest cities in Europe. Additionally, Paris is known for its iconic landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a popular tourist destination, with millions of visitors each year, and its vibrant culture and cuisine make it a must-visit destination for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be a rapidly evolving field, with new technologies and applications continually emerging. Here are some potential trends that may shape the AI landscape in the coming years:
    
    1. Increased Personalization and Adaptability: As AI technology continues to improve, we may see more personalized and adaptable AI systems that learn from user behavior and adapt to new situations in real-time. This could lead to more efficient and effective solutions to a wide range of problems.
    
    2. Increased Transparency and Explainability: As AI systems become more complex and sophisticated, we may see a greater emphasis on increasing transparency and explainability. This could lead to more user-friendly and understandable AI systems


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

     John

     Doe

    .

     I

     am

     an

     experienced

     freelancer

     who

     specializes

     in

     delivering

     high

    -quality

     software

     development

     services

    .

     I

     have

     a

     knack

     for

     finding

     unique

     solutions

     to

     complex

     problems

    ,

     and

     I

     am

     always

     looking

     for

     new

     opportunities

     to

     help

     businesses

     grow

    .

     I

     have

     a

     passion

     for

     working

     with

     clients

     to

     ensure

     they

     achieve

     their

     goals

    ,

     and

     I

     am

     excited

     about

     the

     opportunities

     to

     make

     a

     positive

     impact

     on

     the

     world

    .

     How

     can

     I

     be

     a

     good

     fit

     for

     you

    ?

     Please

     feel

     free

     to

     ask

     me

     any

     questions

     you

     might

     have

    !

     [

    Mark

     your

     messages

     as

     private

    ]

     What

     do

     you

     do

    ?

     What

    's

     your

     favorite

     hobby

    ?


    Hello

    ,

     my

     name

     is

     John

     Doe

    .

     I

     am

     an

     experienced

     freelancer

     who

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Ex

    plain

     your

     answer

    .

     To

     answer

     the

     question

    ,

     consider

     the

     following

    :

     What

     is

     the

     largest

     city

     in

     the

     world

    ?

     The

     largest

     city

     in

     the

     world

     is

     New

     York

     City

    .

     Based

     on

     that

     information

    ,

     we

     can

     conclude

     that

     Paris

     is

     the

     capital

     city

     of

     France

    .

     Therefore

    ,

     the

     answer

     is

     Paris

    .

     Let

     me

     know

     if

     you

     need

     anything

     else

    !

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     significant

     advancements

     in

     various

     areas

    ,

     including

    :
    


    1

    .

     Improved

     accuracy

     and

     precision

    :

     As

     AI

     continues

     to

     learn

     and

     process

     more

     data

    ,

     it

     is

     expected

     to

     become

     even

     more

     accurate

     and

     precise

    ,

     leading

     to

     better

     predictions

     and

     decisions

    .
    


    2

    .

     Enhanced

     natural

     language

     processing

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     understand

     and

     interpret

     human

     language

    ,

     leading

     to

     more

     natural

     and

     intuitive

     interactions

    .
    


    3

    .

     Increased

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     machine

     learning

    ,

     robotics

    ,

     and

     quantum

     computing

    ,

     it

     is

     expected

     to

     have

     a

     more

     significant

     impact

     on

     various

     industries

     and

     applications

    .
    


    4

    .

     Rise

     of

     autonomous

     systems

    :

     As

    



```python
llm.shutdown()
```
