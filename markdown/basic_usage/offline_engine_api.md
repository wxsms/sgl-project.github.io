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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.64it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.54it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:05,  8.17it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:05,  8.17it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:05,  8.17it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:05,  8.17it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:05,  8.17it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:03, 12.16it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:03, 12.16it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:03, 12.16it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:03, 12.16it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:03, 12.16it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 16.31it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 16.31it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 16.31it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 16.31it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:02, 16.31it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:02, 16.31it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:02, 16.31it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:02, 16.31it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:02, 16.31it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:02, 16.31it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:00, 29.08it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:00, 29.08it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:00, 29.08it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:00, 29.08it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:00, 29.08it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:00, 29.08it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:00, 29.08it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:00, 29.08it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:00, 29.08it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:00, 29.08it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:00, 29.08it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 43.16it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 43.16it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 43.16it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 43.16it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 43.16it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 43.16it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 43.16it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 43.16it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 43.16it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 43.16it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 43.16it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 54.84it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 54.84it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 54.84it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 54.84it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 54.84it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 54.84it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 54.84it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 54.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.24 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.87it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.22 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.87it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.22 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.87it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.21 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.21 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.19 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.17 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=960 avail_mem=72.19 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.19 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=832 avail_mem=72.18 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.36it/s]Capturing num tokens (num_tokens=832 avail_mem=72.18 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.36it/s]Capturing num tokens (num_tokens=768 avail_mem=72.18 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.36it/s]Capturing num tokens (num_tokens=704 avail_mem=72.18 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.36it/s]Capturing num tokens (num_tokens=640 avail_mem=72.17 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.36it/s]Capturing num tokens (num_tokens=576 avail_mem=72.17 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.36it/s]Capturing num tokens (num_tokens=512 avail_mem=72.16 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.36it/s]Capturing num tokens (num_tokens=512 avail_mem=72.16 GB):  50%|█████     | 29/58 [00:00<00:00, 42.61it/s]Capturing num tokens (num_tokens=480 avail_mem=72.17 GB):  50%|█████     | 29/58 [00:00<00:00, 42.61it/s]Capturing num tokens (num_tokens=448 avail_mem=72.17 GB):  50%|█████     | 29/58 [00:00<00:00, 42.61it/s]Capturing num tokens (num_tokens=416 avail_mem=72.17 GB):  50%|█████     | 29/58 [00:00<00:00, 42.61it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.17 GB):  50%|█████     | 29/58 [00:00<00:00, 42.61it/s]Capturing num tokens (num_tokens=352 avail_mem=72.16 GB):  50%|█████     | 29/58 [00:00<00:00, 42.61it/s]Capturing num tokens (num_tokens=352 avail_mem=72.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=320 avail_mem=72.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=288 avail_mem=72.15 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=256 avail_mem=72.15 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=240 avail_mem=72.15 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=224 avail_mem=72.14 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=224 avail_mem=72.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=208 avail_mem=72.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=192 avail_mem=72.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=176 avail_mem=72.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.22it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.13 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=144 avail_mem=72.13 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=144 avail_mem=72.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=128 avail_mem=72.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=112 avail_mem=72.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=96 avail_mem=72.12 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s] Capturing num tokens (num_tokens=80 avail_mem=72.12 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=64 avail_mem=72.11 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=64 avail_mem=72.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=48 avail_mem=72.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=32 avail_mem=72.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=28 avail_mem=72.10 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.42it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.10 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=20 avail_mem=72.10 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=20 avail_mem=72.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=16 avail_mem=72.10 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=12 avail_mem=72.09 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=8 avail_mem=72.09 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.60it/s] Capturing num tokens (num_tokens=4 avail_mem=72.08 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.60it/s]Capturing num tokens (num_tokens=4 avail_mem=72.08 GB): 100%|██████████| 58/58 [00:01<00:00, 40.44it/s]


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
    Generated text:  Jennifer, I’m 18 years old, I’m married, and I live with my parents. I’m going to start a job tomorrow, but I’m not sure how to pay for it. Should I go to the bank, get a credit card, or some other money source? Should I go to the bank? How can I get a job? How can I get a job if I’m not sure how to pay for it?
    You can ask for a job interview at the bank, but it doesn’t usually involve a deposit. You have to apply to do a job, so you have to set up an application process
    ===============================
    Prompt: The president of the United States is
    Generated text:  a wealthy man. He lives in a big house with a lot of wealth and money in the bank. He has many important jobs to do. The president lives in a very big house and the president works in important jobs. The president is the leader of the country, that's why many people want to be the president. Every 4 years, the president has to run again. He has to run in elections. The president gets his power from the people by running in elections. The people choose him or her to be the leader of the country. Why can't you be the president? The president has to be elected. But sometimes
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ].
    A. Paris
    B. Brussels
    C. Strasbourg
    D. London
    Answer:
    A
    
    If the ratio of the areas of two similar polygons is 1/8, then their perimeters must be ___.
    A. 1/4
    B. 1/2
    C. 1/8
    D. 1
    Answer:
    A
    
    When using a 250-type level to measure the elevation of a high ground, the scale on the instrument is 1:500. The measured elevation is 150 meters. What is the actual elevation of the ground?
    
    ===============================
    Prompt: The future of AI is
    Generated text:  set to be fully autonomous, without a human input. We are now at a stage where every machine has learned to perform a specific function. These machines are not able to develop their own patterns of thinking and can only learn by directly being instructed to do it. For example, an autonomous vehicle that knows how to navigate a road in a specific area or a robotic arm that is capable of moving on a precise surface like a table. It is clear that the level of complexity of the machines will increase over time, with the further development of artificial intelligence coming to depend on more and more machine learning algorithms. At the same time, the development of


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Name] with [Number] years of experience in [Industry]. I'm a [Number] year old, [Name] with [Number] years of experience in [Industry]. I'm a [Number] year old, [Name] with [Number] years of experience in [Industry]. I'm a [Number] year old, [Name] with [Number] years of experience in [Industry]. I'm a [Number] year old, [Name] with [Number] years of experience in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. The city is home to many cultural institutions, including museums, theaters, and art galleries. Paris is a popular tourist destination, with millions of visitors each year. It is also a major center for business and finance
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are already being used in a wide range of applications, from self-driving cars to personalized medicine to fraud detection. As these technologies continue to improve, we can expect to see even more innovative applications and a greater focus on ethical and social implications. Additionally, there is a growing interest in developing AI that is more transparent and accountable, with greater emphasis on privacy and security. Finally, there is a growing recognition of the importance of AI in driving innovation and economic growth, with more focus on developing AI that is complementary to human capabilities
    


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
    Generated text:  [Name] and I'm an [age] year old [occupation]. I grew up in [city] and graduated from [university]. I'm currently working as a [occupation] at [company]. I enjoy [what you enjoy doing], and I'm really passionate about [what you love doing]. I have a strong sense of [a characteristic of mine], and I believe that [why you believe in yourself]. I'm really looking forward to [something that I'm looking forward to], and I'm excited to make a difference in the world. [You're welcome to make any extra comments or requests]. Good luck with your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, beautiful architecture, and iconic landmarks such as the Eiffel Tower and the Louvre Museum. The city is also home to numerous museums, theaters, and other cultural institutions, making it a popular tourist destination. Paris is a vibrant and exciting city that has become a symbol of French culture and identity. It is one of the largest cities in the world and a major player in global affairs, hosting numerous international events and parades. The city is also home to a unique and diverse population, including French and European immigrants, making it a melting pot of cultures. Overall, Paris is a city that is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and will likely continue to evolve rapidly. Here are some possible trends in AI over the next few decades:
    
    1. Increased focus on ethical considerations: As AI continues to gain more control over our lives, there will be greater emphasis on ethical considerations. This will include issues such as bias, transparency, privacy, and accountability. Governments and businesses will be increasingly implementing regulations and standards to ensure that AI is used responsibly and in ways that benefit society as a whole.
    
    2. Development of more sophisticated AI: The development of AI will continue to improve in accuracy, speed, and efficiency. This will likely result in new AI applications and industries that


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

    Your

     profession

    ]

     who

     has

     always

     been

     fascinated

     by

     [

    What

     you

     do

     for

     a

     living

    ,

     if

     not

     explicitly

     stated

     here

    ].

     I

    'm

     an

     [

    Intro

    verted

     or

     Ext

    ro

    verted

     personality

     type

    ],

     and

     I

     enjoy

     [

    Your

     hobby

     or

     interest

    ].

     I

     have

     always

     been

     determined

     to

     do

     my

     best

    ,

     but

     I

     don

    't

     always

     know

     why

    .

     I

     don

    't

     have

     a

     very

     strong

     communication

     skills

    ,

     but

     I

     like

     to

     listen

     to

     others

     and

     understand

     their

     perspective

    .

     What

     is

     the

     purpose

     of

     your

     introduction

    ?
    


    Sure

    ,

     here

    's

     a

     sample

    :
    


    Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

     I

    'm

     a

     [

    Your

     profession

    ]

     who

     has

     always

     been

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     encaps

    ulates

     the

     main

     facts

     about

     the

     city

    ,

     including

     its

     capital

     status

     and

     its

     name

    .

     It

     is

     straightforward

     and

     easy

     to

     understand

    ,

     making

     it

     suitable

     for

     a

     simple

     text

    .

     Additionally

    ,

     it

     highlights

     the

     significant

     importance

     of

     Paris

     to

     the

     French

     culture

     and

     society

    ,

     making

     it

     a

     good

     starting

     point

     for

     further

     research

     or

     communication

     about

     the

     city

    .

     The

     statement

     is

     concise

     and

     informative

    ,

     making

     it

     suitable

     for

     a

     variety

     of

     contexts

    ,

     from

     literature

     to

     general

     knowledge

    .

     Finally

    ,

     it

     is

     devoid

     of

     any

     potentially

     sensitive

     or

     controversial

     information

    ,

     which

     is

     appropriate

     in

     the

     context

     of

     the

     French

     capital

    .

    
    
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

     that

     will

     shape

     the

     way

     we

     interact

     with

     technology

    ,

     learn

     and

     grow

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     to

     come

     in

     the

     next

     decade

    :
    


    1

    .

     AI

     will

     continue

     to

     improve

     in

     areas

     like

     natural

     language

     processing

     and

     machine

     learning

    .

     We

     are

     already

     seeing

     breakthrough

    s

     in

     chat

    bots

    ,

     voice

     recognition

    ,

     and

     predictive

     analytics

    ,

     but

     we

     also

     have

     the

     potential

     to

     see

     even

     greater

     advancements

     in

     areas

     like

     quantum

     computing

     and

     enhanced

     artificial

     intelligence

    .
    


    2

    .

     AI

     will

     continue

     to

     become

     more

     integrated

     into

     our

     everyday

     lives

    .

     We

     will

     likely

     see

     the

     integration

     of

     AI

     into

     our

     homes

     and

     businesses

    ,

     as

     well

     as

     our

     transportation

     systems

    .

     This

    



```python
llm.shutdown()
```
