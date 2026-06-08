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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:25,  2.02it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:25,  2.02it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:25,  2.02it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:25,  2.02it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:25,  2.02it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:11,  4.10it/s]

    Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:05,  7.39it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:05,  7.39it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:05,  7.39it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:05,  7.39it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:05,  7.39it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:05,  7.39it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:05,  7.39it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:03, 12.20it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:03, 12.20it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:03, 12.20it/s]

    Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:03, 12.20it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:03, 12.20it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:03, 12.20it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:03, 12.20it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:03, 12.20it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:03, 12.20it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 29.78it/s]

    Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 29.78it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 38.88it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 38.88it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 38.88it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 38.88it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 38.88it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 38.88it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 38.88it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 38.88it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 38.88it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 45.62it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 45.62it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 45.62it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 45.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.22 GB):   2%|▏         | 1/58 [00:00<00:05,  9.61it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.18 GB):   2%|▏         | 1/58 [00:00<00:05,  9.61it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=58.18 GB):   3%|▎         | 2/58 [00:00<00:06,  8.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.18 GB):   3%|▎         | 2/58 [00:00<00:06,  8.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.18 GB):   3%|▎         | 2/58 [00:00<00:06,  8.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.18 GB):   7%|▋         | 4/58 [00:00<00:04, 11.12it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.18 GB):   7%|▋         | 4/58 [00:00<00:04, 11.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.17 GB):   7%|▋         | 4/58 [00:00<00:04, 11.12it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=58.16 GB):   7%|▋         | 4/58 [00:00<00:04, 11.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.16 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.16 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.16 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.16 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.01 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.01 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.67it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.00 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.00 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.67it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.00 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.96 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.96 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.63 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.63 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.44 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.63it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=57.44 GB):  31%|███       | 18/58 [00:00<00:01, 24.47it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.44 GB):  31%|███       | 18/58 [00:00<00:01, 24.47it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.44 GB):  31%|███       | 18/58 [00:00<00:01, 24.47it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.42 GB):  31%|███       | 18/58 [00:00<00:01, 24.47it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.42 GB):  36%|███▌      | 21/58 [00:00<00:01, 24.81it/s]Capturing num tokens (num_tokens=960 avail_mem=57.43 GB):  36%|███▌      | 21/58 [00:00<00:01, 24.81it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=57.43 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.81it/s]Capturing num tokens (num_tokens=832 avail_mem=57.42 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.81it/s]Capturing num tokens (num_tokens=832 avail_mem=57.42 GB):  41%|████▏     | 24/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=768 avail_mem=57.42 GB):  41%|████▏     | 24/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=704 avail_mem=57.42 GB):  41%|████▏     | 24/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=640 avail_mem=57.41 GB):  41%|████▏     | 24/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=640 avail_mem=57.41 GB):  47%|████▋     | 27/58 [00:01<00:01, 23.32it/s]Capturing num tokens (num_tokens=576 avail_mem=57.41 GB):  47%|████▋     | 27/58 [00:01<00:01, 23.32it/s]

    Capturing num tokens (num_tokens=512 avail_mem=57.40 GB):  47%|████▋     | 27/58 [00:01<00:01, 23.32it/s]Capturing num tokens (num_tokens=480 avail_mem=57.41 GB):  47%|████▋     | 27/58 [00:01<00:01, 23.32it/s]Capturing num tokens (num_tokens=480 avail_mem=57.41 GB):  52%|█████▏    | 30/58 [00:01<00:01, 24.04it/s]Capturing num tokens (num_tokens=448 avail_mem=57.41 GB):  52%|█████▏    | 30/58 [00:01<00:01, 24.04it/s]Capturing num tokens (num_tokens=416 avail_mem=57.41 GB):  52%|█████▏    | 30/58 [00:01<00:01, 24.04it/s]Capturing num tokens (num_tokens=384 avail_mem=57.41 GB):  52%|█████▏    | 30/58 [00:01<00:01, 24.04it/s]

    Capturing num tokens (num_tokens=384 avail_mem=57.41 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.11it/s]Capturing num tokens (num_tokens=352 avail_mem=57.40 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.11it/s]Capturing num tokens (num_tokens=320 avail_mem=57.40 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.11it/s]Capturing num tokens (num_tokens=288 avail_mem=57.40 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.11it/s]Capturing num tokens (num_tokens=256 avail_mem=57.39 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.11it/s]

    Capturing num tokens (num_tokens=256 avail_mem=57.39 GB):  64%|██████▍   | 37/58 [00:01<00:01, 17.58it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:01, 17.58it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:01, 17.58it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:01, 17.58it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:01, 17.58it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:01, 17.58it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.01it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.01it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.01it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 23.01it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 23.01it/s]

    Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 23.01it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:02<00:00, 27.49it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:02<00:00, 27.49it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:02<00:00, 27.49it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:02<00:00, 27.49it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:02<00:00, 27.49it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:02<00:00, 30.16it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:02<00:00, 30.16it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:02<00:00, 30.16it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 30.16it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 30.16it/s]

    Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 30.16it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.27it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.27it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.27it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:02<00:00, 24.72it/s]


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
    Generated text:  James. My name means "uncle" in the local dialect of Texas. I'm writing a blog post to talk about a recent trip I took to Europe. I'm looking for advice on how to improve my writing skills and improve my ability to communicate with other people. 
    Can you provide me with any advice on how to improve my writing skills and communication skills in general? Additionally, could you please provide some tips on how to write a compelling article or article? 
    Certainly! Here are some general tips for improving your writing skills and communication skills in general:
    1. Read extensively: Reading is a great way to improve your writing skills
    ===============================
    Prompt: The president of the United States is
    Generated text:  a citizen of which country?
    The president of the United States is a citizen of the United States. However, it's worth noting that the president of the United States is not a natural-born citizen, but rather a resident U.S. citizen. The president is elected by the U.S. citizens who are registered to vote and lives in the United States. The president is elected through a process of voting for a specific number of electoral votes, which are distributed to each state. The president is a representative of the people of the United States, and they represent their interests and work to protect and promote the country's interests.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population of Paris is about 2,100,000,000. This number can be written as a decimal number.
    To convert the number 2,100,000,000 into a decimal number, we need to understand the place value of each digit in the number. Each digit represents a power of 10, starting from the rightmost digit (which is the \(10^0\) place) and moving to the left.
    
    The number 2,100,000,000 can be broken down as follows:
    
    1
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the developers of the systems that will be building it. Here are some examples of what the future might hold for AI, and what the people in the technology and business world should do to ensure the future of AI is built with the same standards and rules for developers as we have for the rest of the world. I think it is a good question for the future of AI.
    
    a. The world is going to have a climate with more and more AI applications.
    
    b. AI is going to have more and more physical manifestation.
    
    c. The world is going to have a more crowded world.
    
    d. The world is going


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


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a popular tourist destination and a major economic center. Paris is home to many world-renowned museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. The city is also known for its cuisine, including its famous croissants and its famous French wine. Paris is a vibrant and diverse city with a rich history and culture. Its status as the capital of France has made it a major hub for international trade and diplomacy. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from and improve on their own.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be increased pressure to address ethical concerns related to AI, such as bias, privacy, and transparency. This could lead to more rigorous ethical standards and regulations for AI development and deployment.
    
    3. Increased use of AI for creative tasks:
    


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
    Generated text:  ____________. I'm a(n) _________. I'm _________. I'm an expert in ___________. I have a passion for ___________. What kind of person am I? What do you like about me? What will you like to learn from me? Please tell me what you think about me. Thanks.
    I'm glad you're here. My name is John, and I am a software engineer. I am an expert in software development, and I have a passion for software engineering. I am an expert in developing software for businesses, and I am passionate about sharing my knowledge with others. I like to learn about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, and is located in the center of the country in the Seine River valley. It is the largest city in France, the 10th-largest in the world and the second-largest metropolitan area after New York City in the United States. Paris is also the largest city in the European Union and the second-largest metropolitan area in Europe after London. Its status as the capital is due to its historic importance and many famous landmarks such as the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and the Arc de Triomphe. 
    
    Other significant cities in France include
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  dynamic, and it is difficult to predict exactly what trends will emerge. However, several trends are likely to continue shaping the field in the coming years:
    
    1. Increased integration with human intelligence: One of the biggest trends is the increasing integration of AI with human intelligence. AI systems will become more sophisticated and able to learn and adapt based on real-world data and feedback. This will enable AI systems to better understand human emotions, preferences, and behaviors, and to make more accurate predictions and recommendations.
    
    2. Enhanced ethical AI: The use of AI for ethical purposes will become more common. As AI systems become more sophisticated, they will be able to


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

    ],

     a

     professional

     writer

     with

     over

     [

    X

    ]

     years

     of

     experience

     in

     the

     industry

    .

     I

     specialize

     in

     [

    Your

     Specialty

    ]

     and

     have

     won

     several

     awards

     for

     my

     work

    .

     What

     can

     I

     say

     to

     a

     potential

     employer

     or

     client

    ?

     G

    reetings

    !

     My

     name

     is

     [

    Your

     Name

    ],

     a

     professional

     writer

     with

     over

     [

    X

    ]

     years

     of

     experience

     in

     the

     industry

    .

     I

     specialize

     in

     [

    Your

     Specialty

    ]

     and

     have

     won

     several

     awards

     for

     my

     work

    .

     What

     can

     I

     say

     to

     a

     potential

     employer

     or

     client

    ?

     Cheers

    !

     [

    Your

     Name

    ]

     Looking

     forward

     to

     your

     day

    !

     How

     do

     you

     do

    ?

     [

    Your

     Name

    ]

     Looking

     forward

     to

     your

     day

    !

     Thanks

     for

     the

    
    
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

     Europe

     and

     one

     of

     the

     largest

     cities

     in

     the

     world

    .

     It

     is

     known

     for

     its

     historical

     landmarks

    ,

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

     French

     cuisine

     and

     fashion

     industry

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     hub

     for

     business

     and

     commerce

    .

     It

     has

     a

     rich

     history

    ,

     culture

    ,

     and

     diverse

     population

    .

     With

     its

     

    1

     million

     residents

    ,

     Paris

     is

     the

     heart

     of

     the

     French

    -speaking

     world

    .

     Its

     reputation

     as

     the

     "

    city

     of

     love

    "

     and

     "

    city

     of

     light

    "

     makes

     it

     an

     important

     part

     of

     French

     national

     identity

    .

     The

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     involve

     significant

     advancements

     in

     several

     key

     areas

    ,

     including

    :
    


    1

    .

     Increased

     Focus

     on

     Eth

    ical

     and

     Social

     Concern

    s

    :

     The

     AI

     industry

     is

     already

     grappling

     with

     issues

     such

     as

     bias

     and

     discrimination

     in

     decision

    -making

    .

     As

     AI

     systems

     become

     more

     prevalent

    ,

     there

     will

     be

     a

     greater

     focus

     on

     developing

     ethical

     and

     social

     principles

     for

     their

     development

     and

     use

    .
    


    2

    .

     Integration

     of

     AI

     with

     Physical

     and

     Biological

     Sciences

    :

     As

     AI

     becomes

     more

     integrated

     with

     various

     physical

     and

     biological

     sciences

    ,

     there

     will

     be

     an

     increased

     emphasis

     on

     creating

     systems

     that

     can

     accurately

     predict

     and

     model

     the

     behavior

     of

     these

     systems

    .
    


    3

    .

     Development

     of

     AI

     with

     Human

    -like

     Intelligence

    :

     AI

     systems

     are

     expected

     to

     continue

     developing

     with

    



```python
llm.shutdown()
```
