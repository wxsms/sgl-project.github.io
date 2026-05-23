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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:00,  5.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:00,  5.28s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:00,  5.28s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:00,  5.28s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:00,  5.28s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:42,  1.24it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.36it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:05,  7.58it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:05,  7.58it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:05,  7.58it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:05,  7.58it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:05,  7.58it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:05,  7.58it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:05,  7.58it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:05,  7.58it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:05,  7.58it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.15it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.15it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.15it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.15it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.15it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.15it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.15it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:01, 15.98it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:01, 15.98it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:01, 15.98it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:01, 15.98it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:01, 15.98it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:01, 15.98it/s]

    Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:01, 15.98it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:01, 15.98it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:05<00:01, 15.98it/s]Compiling num tokens (num_tokens=160):  59%|█████▊    | 34/58 [00:05<00:01, 15.98it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 23.62it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 23.62it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 23.62it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 23.62it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 23.62it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 23.62it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:06<00:00, 23.62it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:06<00:00, 23.62it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:06<00:00, 23.62it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:06<00:00, 23.62it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:06<00:00, 31.93it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:06<00:00, 31.93it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:06<00:00, 31.93it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:06<00:00, 31.93it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:06<00:00, 31.93it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:06<00:00, 31.93it/s] 

    Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:06<00:00, 31.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.58it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.58it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.58it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.75 GB):   7%|▋         | 4/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.75 GB):   7%|▋         | 4/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.75 GB):   7%|▋         | 4/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.75 GB):  10%|█         | 6/58 [00:00<00:03, 14.95it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.25 GB):  10%|█         | 6/58 [00:00<00:03, 14.95it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.24 GB):  10%|█         | 6/58 [00:00<00:03, 14.95it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=76.24 GB):  10%|█         | 6/58 [00:00<00:03, 14.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:03, 14.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.72it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 29.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 29.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 29.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  31%|███       | 18/58 [00:00<00:01, 29.17it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  31%|███       | 18/58 [00:00<00:01, 29.17it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  31%|███       | 18/58 [00:00<00:01, 29.17it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  40%|███▉      | 23/58 [00:00<00:01, 33.26it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:01, 33.26it/s]

    Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:01, 33.26it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:01, 33.26it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  40%|███▉      | 23/58 [00:00<00:01, 33.26it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.52it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.52it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.52it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.52it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.52it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.05it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.05it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.05it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.05it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.05it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  60%|██████    | 35/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.61it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.61it/s]

    Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.61it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.61it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 34.61it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 35.28it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 33.88it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 33.88it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 33.88it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 33.88it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 33.88it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.77it/s]

    Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.32it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 31.25it/s]


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
    Generated text:  Meirion Elroch, and I am the Keeper of the Broken Isles. As a Keeper, I take care of the state of the Broken Isles, including all the people and animals within the island, and I am responsible for maintaining peace and tranquility on the island. I am also responsible for making sure that the Broken Isles is safe and secure for all of the people and animals who live there.
    My name is meant to be a reminder that I am a Keeper of the Broken Isles, and that I am the guardian of the island's inhabitants and the people who live there. I am also a Keeper who is committed to maintaining
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He helps to make the country a better place for all people. He is in charge of the government and the army, as well as other important jobs. He has a very big job because he is the president of the United States. He is very busy, but he is very happy. He makes sure that there are no bad things or no accidents. The president of the United States also wants to help people in the country. He wants to make sure that there are no serious problems. The president's job is to help his country, and he always works hard to make his country a better place to live.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest and most populous city of France. The population of Paris as of 2005 was 2.25 million. It is located in the western part of the country in the region of Île-de-France.
    
    Based on that paragraph can we conclude that this sentence is true?
    Paris is not a city.
    
    Choose from:
     A). yes.
     B). no.
    
    B). no.
    
    The paragraph clearly states that Paris is the capital of France and is located in the western part of France, making it a city. Therefore, we cannot conclude that Paris is not a city based on the given information
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and unpredictable, and there are significant gaps in our understanding of it. As AI continues to advance, it has the potential to revolutionize many industries and improve our lives. However, we should also be aware of the potential risks and challenges that come with AI. In this report, we will explore the current state of AI research and its impact on society, as well as discuss the challenges and opportunities that lie ahead. We will also consider the ethical implications of AI and the potential impact it could have on different sectors of society. Finally, we will provide some recommendations for how individuals and organizations can stay ahead of the curve in the world of


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also famous for its fashion industry, art scene, and cuisine. Paris is a vibrant and diverse city with a rich cultural and artistic heritage. It is a popular tourist destination and a major economic center in Europe. The city is home to many world-renowned museums, theaters, and other cultural institutions. Paris is a city of contrasts, with its modern
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud detection,
    


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
    Generated text:  [Name], and I'm a [job title] who is passionate about [what I do for a living]. I'm a [number] in the [category] of [job title]. I enjoy [why I do what I do] and [any other interests you might have]. I'm always looking for opportunities to learn new skills and expand my knowledge, so I'm always eager to [tell a fun fact or have a conversation]. I'm very [attentive, friendly, optimistic, etc.] and enjoy [why I'm good at what I do]. I value [my values or beliefs], and I strive to [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A. True
    B. False
    
    B. False
    
    Paris is not the capital of France. It is the capital of the country. The capital of France is Paris. France's capital is Paris. France's capital is Paris. The capital of France is Paris. Paris is the capital of France. The capital of France is Paris. Paris is the capital of France. The capital of France is Paris. Paris is the capital of France. Paris is the capital of France. Paris is the capital of France. Paris is the capital of France. Paris is the capital of France. Paris is the capital of France. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  one of rapid technological and societal change, with the potential for dramatic changes in areas such as personal assistants, autonomous vehicles, and the development of advanced cognitive technologies. Here are some possible trends in AI that we can expect to see in the coming years:
    
    1. Increased automation of tasks: As AI continues to improve, we can expect to see an increase in the automation of tasks that are currently done by humans, such as data entry, customer service, and administrative tasks. This could lead to new job roles and the creation of entirely new industries.
    
    2. Personalization and prediction: AI is now able to learn from the data it processes,


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

    Age

    ]

     year

     old

     [

    Career

    ].

     I

     love

     [

    What

     You

     Do

    ]

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     [

    What

     You

     Do

    ].

     I

    'm

     [

    Any

     Extra

     Info

    ]

     about

     [

    Age

    ].

     What

     brings

     you

     to

     this

     world

    ?

     To

     be

     honest

    ,

     I

    've

     always

     been

     a

     [

    Your

     Hobby

    ,

     Interest

    ,

     or

     Passion

    ].

     But

     [

    Any

     Extra

     Info

    ]

     about

     [

    Your

     Hobby

    ,

     Interest

    ,

     or

     Passion

    ]

     has

     always

     been

     one

     of

     my

     biggest

     passions

    .

     I

    've

     always

     been

     curious

     to

     learn

     more

     about

     everything

    ,

     and

     [

    Any

     Extra

     Info

    ]

     about

     [

    Your

     Hobby

    ,

     Interest

    ,

     or

     Passion

    ]

     has

     been

     one

     of

    
    
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

    ,

     with

     an

     estimated

     population

     of

     over

     

    2

     million

     people

    ,

     and

     is

     known

     for

     its

     rich

     history

     and

     beautiful

     architecture

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

     The

     city

     is

     also

     famous

     for

     its

     food

     and

     drink

    ,

     as

     well

     as

     its

     fashion

     industry

    .

     Paris

     has

     a

     rich

     cultural

     heritage

    ,

     with

     important

     museums

    ,

     galleries

    ,

     and

     theaters

    .

     It

     is

     also

     a

     bustling

     destination

     for

     tourists

     and

     exp

    ats

     alike

    .

     
    


    Paris

     is

     a

     modern

     city

     with

     a

     strong

     sense

     of

     French

     identity

     and

     culture

    ,

     and

     has

     been

     an

     important

     center

     of

     political

    ,

     economic

    ,

     and

     cultural

     activity

     in

     France

     for

     centuries

    .

     It

     has

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     poised

     to

     evolve

     in

     many

     exciting

     ways

    ,

     bringing

     about

     a

     range

     of

     new

     technologies

     and

     applications

     that

     will

     transform

     the

     way

     we

     live

    ,

     work

    ,

     and

     interact

     with

     the

     world

     around

     us

    .

     Here

     are

     some

     of

     the

     key

     trends

     that

     are

     likely

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Automation

     and

     Artificial

     Intelligence

     (

    AI

    ):

     This

     is

     the

     most

     visible

     and

     directly

     observable

     trend

     of

     AI

    ,

     and

     it

     refers

     to

     the

     use

     of

     machines

     to

     perform

     tasks

     that

     typically

     require

     human

     intelligence

    ,

     such

     as

     data

     analysis

    ,

     decision

    -making

    ,

     and

     problem

    -solving

    .

     Automation

     will

     increase

     productivity

    ,

     reduce

     costs

    ,

     and

     improve

     efficiency

    ,

     leading

     to

     a

     more

     productive

     and

     competitive

     society

    .
    


    2

    .

     Deep

     Learning

     and

    



```python
llm.shutdown()
```
