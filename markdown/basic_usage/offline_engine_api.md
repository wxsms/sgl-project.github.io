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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.44it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]

    Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:03, 10.42it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:03, 10.42it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:03, 10.42it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:03, 10.42it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:03, 10.42it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:03, 10.42it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:03, 10.42it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:03, 10.42it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 15.84it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 15.84it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 15.84it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 15.84it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 15.84it/s]

    Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 15.84it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 15.84it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:01, 15.84it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 21.95it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 21.95it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 21.95it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 21.95it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 21.95it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 21.95it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 21.95it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 21.95it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 28.49it/s]

    Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 28.49it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 35.15it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 35.15it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 35.15it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 35.15it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 35.15it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 35.15it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 35.15it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 35.15it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 35.15it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 43.24it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 43.24it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.71 GB):   2%|▏         | 1/58 [00:00<00:05,  9.88it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.67 GB):   2%|▏         | 1/58 [00:00<00:05,  9.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.67 GB):   2%|▏         | 1/58 [00:00<00:05,  9.88it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=55.67 GB):   5%|▌         | 3/58 [00:00<00:05, 10.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.67 GB):   5%|▌         | 3/58 [00:00<00:05, 10.87it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.67 GB):   5%|▌         | 3/58 [00:00<00:05, 10.87it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.67 GB):   9%|▊         | 5/58 [00:00<00:04, 12.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.66 GB):   9%|▊         | 5/58 [00:00<00:04, 12.19it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=55.65 GB):   9%|▊         | 5/58 [00:00<00:04, 12.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.65 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.65 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.65 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.65 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.65 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.34it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=55.64 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.64 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.64 GB):  21%|██        | 12/58 [00:00<00:02, 17.69it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.63 GB):  21%|██        | 12/58 [00:00<00:02, 17.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.63 GB):  21%|██        | 12/58 [00:00<00:02, 17.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.63 GB):  21%|██        | 12/58 [00:00<00:02, 17.69it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=55.63 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.92it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.63 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.92it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.62 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.62 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.62 GB):  31%|███       | 18/58 [00:01<00:01, 22.26it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.62 GB):  31%|███       | 18/58 [00:01<00:01, 22.26it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.61 GB):  31%|███       | 18/58 [00:01<00:01, 22.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.59 GB):  31%|███       | 18/58 [00:01<00:01, 22.26it/s]Capturing num tokens (num_tokens=960 avail_mem=55.61 GB):  31%|███       | 18/58 [00:01<00:01, 22.26it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=55.61 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.52it/s]Capturing num tokens (num_tokens=896 avail_mem=55.61 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.52it/s]Capturing num tokens (num_tokens=832 avail_mem=55.60 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.52it/s]Capturing num tokens (num_tokens=768 avail_mem=55.60 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.52it/s]Capturing num tokens (num_tokens=704 avail_mem=55.60 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.52it/s]Capturing num tokens (num_tokens=704 avail_mem=55.60 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.94it/s]Capturing num tokens (num_tokens=640 avail_mem=55.59 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.94it/s]Capturing num tokens (num_tokens=576 avail_mem=55.59 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.94it/s]Capturing num tokens (num_tokens=512 avail_mem=55.58 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.94it/s]

    Capturing num tokens (num_tokens=480 avail_mem=55.59 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.94it/s]Capturing num tokens (num_tokens=480 avail_mem=55.59 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.64it/s]Capturing num tokens (num_tokens=448 avail_mem=55.59 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.64it/s]Capturing num tokens (num_tokens=416 avail_mem=55.59 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.64it/s]Capturing num tokens (num_tokens=384 avail_mem=55.59 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.64it/s]Capturing num tokens (num_tokens=352 avail_mem=55.58 GB):  52%|█████▏    | 30/58 [00:01<00:00, 29.64it/s]Capturing num tokens (num_tokens=352 avail_mem=55.58 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.99it/s]Capturing num tokens (num_tokens=320 avail_mem=55.58 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.99it/s]Capturing num tokens (num_tokens=288 avail_mem=55.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.99it/s]

    Capturing num tokens (num_tokens=256 avail_mem=55.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.99it/s]Capturing num tokens (num_tokens=240 avail_mem=55.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.99it/s]Capturing num tokens (num_tokens=240 avail_mem=55.57 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.40it/s]Capturing num tokens (num_tokens=224 avail_mem=55.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.40it/s]Capturing num tokens (num_tokens=208 avail_mem=55.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.40it/s]Capturing num tokens (num_tokens=192 avail_mem=55.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.40it/s]Capturing num tokens (num_tokens=176 avail_mem=55.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.40it/s]Capturing num tokens (num_tokens=176 avail_mem=55.56 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.26it/s]Capturing num tokens (num_tokens=160 avail_mem=55.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.26it/s]Capturing num tokens (num_tokens=144 avail_mem=55.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.26it/s]

    Capturing num tokens (num_tokens=128 avail_mem=55.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.26it/s]Capturing num tokens (num_tokens=112 avail_mem=55.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.26it/s]Capturing num tokens (num_tokens=112 avail_mem=55.55 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.79it/s]Capturing num tokens (num_tokens=96 avail_mem=55.54 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.79it/s] Capturing num tokens (num_tokens=80 avail_mem=55.54 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.79it/s]Capturing num tokens (num_tokens=64 avail_mem=55.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.79it/s]Capturing num tokens (num_tokens=48 avail_mem=55.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 33.79it/s]Capturing num tokens (num_tokens=48 avail_mem=55.53 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.09it/s]Capturing num tokens (num_tokens=32 avail_mem=55.53 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.09it/s]

    Capturing num tokens (num_tokens=28 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.09it/s]Capturing num tokens (num_tokens=24 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:02<00:00, 34.09it/s]Capturing num tokens (num_tokens=20 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:02<00:00, 34.09it/s]Capturing num tokens (num_tokens=20 avail_mem=55.52 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.04it/s]Capturing num tokens (num_tokens=16 avail_mem=55.52 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.04it/s]Capturing num tokens (num_tokens=12 avail_mem=55.51 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.04it/s]Capturing num tokens (num_tokens=8 avail_mem=55.51 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.04it/s] Capturing num tokens (num_tokens=4 avail_mem=55.50 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.04it/s]Capturing num tokens (num_tokens=4 avail_mem=55.50 GB): 100%|██████████| 58/58 [00:02<00:00, 34.44it/s]Capturing num tokens (num_tokens=4 avail_mem=55.50 GB): 100%|██████████| 58/58 [00:02<00:00, 26.61it/s]


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
    Generated text:  Lee, and I'm a high school student. I'm good at playing the guitar and I'm really interested in it. The guitar is my favorite instrument. I think the guitar is very useful and I want to learn it. However, I have problems with the guitar. First, I can't play the guitar. Second, I don't know how to buy a guitar. Last, I don't know where to find the right guitar. What can I do to learn the guitar? I'm a bit afraid of trying out myself. Do you have any good advice for me?
    Thank you very much for your help. Lee. �
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to increase the number of women in executive positions. The current ratio of women to men in the White House is 3:5. If the president decides to increase the number of women to 30% of the current number, how many women will there be in the White House in total?
    To determine the number of women in the White House after the president decides to increase the number of women to 30% of the current number, we need to follow these steps:
    
    1. **Identify the current ratio of women to men in the White House:**
       The current ratio of women to men is given as 
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. This is the city that is the oldest of Europe’s capitals. It’s the birthplace of the French Revolution and the symbol of the nation. It’s also the seat of power for the first four presidents of the republic.
    But Paris, like many cities, is not static. With over 17 million residents, it’s an urban city with a growing population. The city of Paris is facing a major challenge: the pressure of urban sprawl, of population growth, and the effects of climate change.
    To better understand the growth and dynamics of Paris, we spoke to Clément Marquez, a geographer. Clément
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of you and your brain, which is responsible for processing and storing memories. However, your brain also has limitations such as a limited ability to store and retrieve large amounts of information. Therefore, the most effective way to improve your abilities is to use a combination of both memory and processing capabilities, rather than focusing solely on one aspect. To make the most of your brain, you should continue to engage in a variety of activities that challenge your cognitive abilities, such as puzzles and brain teasers, learning new things, and engaging in mental exercises. Additionally, making use of the natural human responses, such as language and body language,


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] with a passion for [Interest]. I'm a [Skill] with a passion for [Interest]. I'm a [Skill] with a passion for [Interest]. I'm a [Skill] with a passion for [Interest]. I'm a [Skill] with a passion for [Interest]. I'm a [Skill] with a passion for [Interest]. I'm a [Skill] with a passion for [Interest]. I'm a [Skill] with a passion for [Interest]. I'm a [Skill] with a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and restaurants. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of French colonialism and the influence of the French Revolution. It is also home to many notable French artists, writers, and musicians. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to inspire and captivate people around the world. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations. This integration could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability. AI developers will need to be more
    


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
    Generated text:  [Name], I'm a [Job or Profession] who was born in [City] and currently residing in [City]. I have been an avid reader for [Number] years, and I enjoy using my skills in [Skill or Profession] to help people. I’m a [Type] person, comfortable with both the written and verbal communication, and always strive to be a good listener. I’m always looking for ways to improve myself, and I’m always eager to learn new things. I’m a [Type] person, easily adaptable and adaptable to different situations, and I’m always looking for ways to make the world a better
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located on the island of France and served as the official seat of government until the French Revolution in 1789.
    
    The capital of France is Paris, located on the island of France and served as the official seat of government until the French Revolution in 1789. The city, known as "La Grande Reine" (Queen of the Great City), is home to the National Library, the Louvre Museum, the Eiffel Tower, and many other notable landmarks. It has been a center of French culture and history for over 400 years and continues to be a major city in France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a wide range of trends and developments, including:
    
    1. Increased automation and robotics: AI will become more sophisticated and widespread, with the ability to perform tasks that are currently done by humans.
    
    2. Enhanced natural language processing: AI will become better at understanding and generating human-like speech and language, making interactions with humans more natural and intuitive.
    
    3. Improved image and video recognition: AI will become more accurate and capable of recognizing and processing images and videos, from surveillance systems to facial recognition applications.
    
    4. Better autonomous systems: AI will be able to operate autonomous systems, from drones to self-driving cars, which will be able


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

    ,

     and

     I

    'm

     a

     professional

     lingu

    ist

     with

     over

     

    1

    5

     years

     of

     experience

    .

     I

     specialize

     in

     helping

     people

     improve

     their

     language

     skills

     through

     various

     methods

     such

     as

     language

     learning

    ,

     grammar

     analysis

    ,

     and

     communication

     skills

     enhancement

    .
    


    I

    'm

     passionate

     about

     using

     my

     knowledge

     and

     skills

     to

     help

     people

     succeed

     in

     their

     language

     pursuits

    .

     Whether

     it

    's

     improving

     my

     own

     language

     abilities

     or

     helping

     others

     improve

     their

     own

    ,

     I

    'm

     always

     looking

     for

     new

     ways

     to

     learn

     and

     grow

    .
    


    I

    'm

     also

     a

     lover

     of

     literature

    ,

     music

    ,

     and

     photography

    .

     I

     enjoy

     exploring

     new

     worlds

     through

     these

     mediums

    ,

     and

     I

    'm

     always

     on

     the

     lookout

     for

     the

     next

     literary

     mystery

     or

     a

     photo

     that

     will

     make

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     known

     for

     its

     rich

     history

     and

     beautiful

     architecture

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     many

     famous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     known

     for

     its

     vibrant

     culture

    ,

     food

    ,

     and

     wine

    ,

     as

     well

     as

     its

     annual

     French

     Festival

     of

     Flowers

     and

     the

     annual

     E

    iff

    el

     Tower

     Festival

    .

     The

     city

    's

     unique

     blend

     of

     French

     culture

     and

     modern

    ity

     has

     made

     it

     a

     popular

     tourist

     destination

     for

     many

     years

    .

     Paris

     is

     a

     must

    -

    visit

     destination

     for

     anyone

     looking

     to

     experience

     the

     city

    's

     rich

     history

    ,

     beauty

    ,

     and

     culture

    .

     

     Other

     famous

     cities

     in

     France

     include

     Lyon

    ,

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     be

     shaped

     by

     a

     number

     of

     trends

    ,

     including

    :
    


     

     

    1

    .

     Increased

     emphasis

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     likely

     be

     increased

     scrutiny

     of

     its

     use

     and

     development

    .

     As

     a

     result

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     ensuring

     that

     AI

     is

     developed

     and

     used

     in

     a

     responsible

     and

     ethical

     manner

    .


     

     

    2

    .

     Greater

     focus

     on

     automation

     and

     robotics

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     it

     is

     likely

     to

     have

     a

     significant

     impact

     on

     the

     way

     we

     work

     and

     interact

     with

     the

     world

    .

     As

     a

     result

    ,

     there

     may

     be

     a

     greater

     emphasis

     on

     developing

     and

     using

     robots

     and

     automation

    



```python
llm.shutdown()
```
