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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]

    Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:03, 10.39it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:03, 10.39it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:03, 10.39it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:03, 10.39it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:03, 10.39it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:03, 10.39it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:03, 10.39it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:03, 10.39it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:03, 10.39it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:03, 10.39it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 17.62it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 17.62it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 17.62it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 17.62it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 17.62it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 17.62it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 17.62it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 17.62it/s]

    Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 17.62it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 17.62it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 25.89it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 25.89it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 34.87it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 34.87it/s]

    Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 34.87it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 43.94it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 43.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.51it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.34 GB):   3%|▎         | 2/58 [00:00<00:04, 12.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.33 GB):   3%|▎         | 2/58 [00:00<00:04, 12.52it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.33 GB):   3%|▎         | 2/58 [00:00<00:04, 12.52it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.33 GB):   7%|▋         | 4/58 [00:00<00:04, 13.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.28 GB):   7%|▋         | 4/58 [00:00<00:04, 13.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.31 GB):   7%|▋         | 4/58 [00:00<00:04, 13.07it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.31 GB):  10%|█         | 6/58 [00:00<00:03, 14.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.30 GB):  10%|█         | 6/58 [00:00<00:03, 14.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.29 GB):  10%|█         | 6/58 [00:00<00:03, 14.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.27 GB):  10%|█         | 6/58 [00:00<00:03, 14.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.27 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.26 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.23it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.26 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.23it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.25 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.25 GB):  21%|██        | 12/58 [00:00<00:02, 19.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.24 GB):  21%|██        | 12/58 [00:00<00:02, 19.51it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.25 GB):  21%|██        | 12/58 [00:00<00:02, 19.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.26 GB):  21%|██        | 12/58 [00:00<00:02, 19.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.26 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.23 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.76it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.25 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.76it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=74.24 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.24 GB):  31%|███       | 18/58 [00:00<00:01, 23.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.21 GB):  31%|███       | 18/58 [00:00<00:01, 23.46it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.22 GB):  31%|███       | 18/58 [00:00<00:01, 23.46it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.20 GB):  31%|███       | 18/58 [00:00<00:01, 23.46it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.20 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.37it/s]Capturing num tokens (num_tokens=960 avail_mem=74.21 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.37it/s] Capturing num tokens (num_tokens=896 avail_mem=74.20 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.37it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.20 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.37it/s]Capturing num tokens (num_tokens=832 avail_mem=74.20 GB):  41%|████▏     | 24/58 [00:01<00:01, 24.97it/s]Capturing num tokens (num_tokens=768 avail_mem=74.19 GB):  41%|████▏     | 24/58 [00:01<00:01, 24.97it/s]Capturing num tokens (num_tokens=704 avail_mem=74.19 GB):  41%|████▏     | 24/58 [00:01<00:01, 24.97it/s]Capturing num tokens (num_tokens=640 avail_mem=74.18 GB):  41%|████▏     | 24/58 [00:01<00:01, 24.97it/s]Capturing num tokens (num_tokens=640 avail_mem=74.18 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.72it/s]Capturing num tokens (num_tokens=576 avail_mem=74.18 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.72it/s]Capturing num tokens (num_tokens=512 avail_mem=74.16 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.72it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.17 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.72it/s]Capturing num tokens (num_tokens=480 avail_mem=74.17 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.13it/s]Capturing num tokens (num_tokens=448 avail_mem=74.17 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.13it/s]Capturing num tokens (num_tokens=416 avail_mem=74.16 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.13it/s]Capturing num tokens (num_tokens=384 avail_mem=74.16 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.13it/s]Capturing num tokens (num_tokens=384 avail_mem=74.16 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.85it/s]Capturing num tokens (num_tokens=352 avail_mem=74.15 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.85it/s]Capturing num tokens (num_tokens=320 avail_mem=74.14 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.85it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.13 GB):  57%|█████▋    | 33/58 [00:01<00:00, 26.85it/s]Capturing num tokens (num_tokens=288 avail_mem=74.13 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.59it/s]Capturing num tokens (num_tokens=256 avail_mem=74.13 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.59it/s]Capturing num tokens (num_tokens=240 avail_mem=74.12 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.59it/s]Capturing num tokens (num_tokens=224 avail_mem=74.12 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.59it/s]Capturing num tokens (num_tokens=224 avail_mem=74.12 GB):  67%|██████▋   | 39/58 [00:01<00:00, 26.58it/s]Capturing num tokens (num_tokens=208 avail_mem=74.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 26.58it/s]Capturing num tokens (num_tokens=192 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 26.58it/s]

    Capturing num tokens (num_tokens=176 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 26.58it/s]Capturing num tokens (num_tokens=176 avail_mem=74.10 GB):  72%|███████▏  | 42/58 [00:01<00:00, 26.52it/s]Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 26.52it/s]Capturing num tokens (num_tokens=144 avail_mem=74.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 26.52it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 26.52it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 26.91it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 26.91it/s]Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 26.91it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:02<00:00, 26.91it/s]Capturing num tokens (num_tokens=80 avail_mem=74.08 GB):  83%|████████▎ | 48/58 [00:02<00:00, 23.97it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  83%|████████▎ | 48/58 [00:02<00:00, 23.97it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  83%|████████▎ | 48/58 [00:02<00:00, 23.97it/s]Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  83%|████████▎ | 48/58 [00:02<00:00, 23.97it/s]Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:02<00:00, 23.12it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:02<00:00, 23.12it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:02<00:00, 23.12it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:02<00:00, 23.12it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:02<00:00, 23.12it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.75it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.75it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.75it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.75it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:02<00:00, 24.04it/s]


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
    Generated text:  Andrew. I'm 21 years old and live in New York. I have been working in New York for over 3 years. I enjoy hiking, reading, and working out. I have been married for 4 years. My wife, who is an Englishman, has been a teacher for over 10 years. We have two children, and one of us is 55 years old. I think it is only a short time away before our children grow up. We are a very happy couple. 
    
    What was the most interesting thing you did this year?
    
    I have been reading books about the culture of the Roman
    ===============================
    Prompt: The president of the United States is
    Generated text:  invited to a gathering of 1200 people. 40% of the guests are women, 30% are men, and the rest are children. How many children are at the gathering? To determine the number of children attending the gathering, we need to calculate 60% of the total number of guests, since 40% are women and the rest are children.
    
    First, we calculate the number of women:
    \[ 40\% \text{ of } 1200 = 0.40 \times 1200 = 480 \]
    
    Next,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Does it follow that "Paris is the capital of France."?
    Pick your answer from:
    [+] yes
    [+] it is not possible to tell
    [+] no
    
    yes
    ===============================
    Prompt: The future of AI is
    Generated text:  a major topic of discussion for many people. The technological advancements in AI have been making the world more prosperous and efficient. But the questions that arise also become more complex as AI continues to evolve.
    The creation of a world where AI and humans work together is an exciting prospect, but it also poses a challenge to ethical considerations. In order to ensure that AI is used ethically, there are several factors that need to be considered.
    In this article, we will explore the ethical implications of AI and explore the different perspectives on how AI should be used. We will also discuss the different types of ethical considerations involved in AI and the potential challenges that


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. 
    
    (Note: The statement should be a single sentence and should not include any personal opinions or subjective information.) Paris is the capital city of France, renowned for its iconic Eiffel Tower, Notre-Dame Cathedral, and vibrant cultural scene. 
    
    (Note: The statement should be a single sentence and should not include any personal opinions or subjective information.) Paris is the capital city of France, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. 
    
    (Note: The statement should be a single sentence and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This could lead to more sophisticated and nuanced AI that can better understand and respond to human emotions and behaviors.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, it is likely to be used even more extensively in healthcare,
    


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
    Generated text:  [Your Name], and I am a [Your Role/Position] at [Your Organization/Organization Name]. In your time at [Your Organization/Organization Name], I have had the opportunity to grow as a leader, mentor, and team player. I have worked with teams of all sizes, from small startups to large corporations. I have also had the pleasure of learning from experienced leaders and trainers, and I strive to always strive to improve and grow as a professional. I am confident in my abilities and ready to take on new challenges. Thank you. You are now the [Your Name]. How can I assist you today? Start the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is located in the Île de la Cité and the site of the Eiffel Tower. Paris is the fifth largest city in the European Union and the largest French city by population. The city is known for its rich history, beautiful architecture, and annual cultural festivals such as the World Cup and the Musée d'Orsay. It is also home to the United Nations, French National Radio and Television, and the Louvre Museum. Paris's cultural and educational institutions, such as the Louvre Museum and the UNESCO World Heritage Site, are celebrated worldwide. France's capital has been the seat of government, the seat
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but it is likely to continue to evolve and change over the years. Here are some potential future trends in AI:
    
    1. Increased diversity and inclusion: As more people become interested in AI, the technology will become more diverse and inclusive. There will be greater representation from different backgrounds and cultures, and the technology will be more transparent and fair.
    
    2. Integration with everyday life: AI will continue to become more integrated with everyday life, from smartphones and wearables to robots and autonomous vehicles. This will lead to a more connected and connected society.
    
    3. Personalized experiences: AI will become more personal, allowing for more customized and personalized


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

    ]

     and

     I

     am

     a

     [

    age

    ]

     year

     old

     [

    gender

    ]

     who

     is

     [

    occupation

    ].

     I

     am

     currently

     [

    location

    ]

     and

     have

     [

    number

    ]

     years

     of

     experience

    .

     I

     enjoy

     [

    occupation

    ]

     and

     spend

     [

    number

    ]

     hours

     per

     week

     exercising

    ,

     [

    number

    ]

     hours

     per

     week

     studying

    ,

     and

     [

    number

    ]

     hours

     per

     week

     sleeping

    .

     I

     like

     [

    food

    ],

     [

    drink

    ],

     and

     [

    animal

    (s

    )].

     I

     am

     [

    gender

    ]

     and

     [

    gender

    ]

     and

     I

     am

     always

     [

    cur

    iosity

     level

    ]

     when

     it

     comes

     to

     learning

     new

     things

    .

     I

     love

     to

     [

    job

    /h

    obby

    ]

     and

     [

    job

    /h

    obby

    ]

     is

     [

    reason

     for

     interest

    ].

     I

    
    
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

     Love

    .

     It

     is

     a

     cosm

    opolitan

     city

     with

     a

     rich

     history

     and

     a

     lively

     culture

    .

     Paris

     is

     renowned

     for

     its

     art

    ,

     music

    ,

     and

     cuisine

    ,

     and

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

     The

     city

     also

     hosts

     world

    -ren

    owned

     festivals

     and

     events

     throughout

     the

     year

    .

     Paris

     has

     a

     diverse

     population

     and

     cultural

     heritage

     that

     has

     played

     a

     significant

     role

     in

     shaping

     French

     identity

     and

     society

    .

     In

     short

    ,

     Paris

     is

     a

     vibrant

    ,

     romantic

    ,

     and

     beloved

     city

     that

     has

     a

     long

     history

     and

     continues

     to

     be

     a

     global

     center

     of

     culture

     and

     commerce

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     varied

    ,

     with

     potential

     applications

     ranging

     from

     self

    -driving

     cars

     and

     smart

     homes

     to

     personalized

     medicine

     and

     predictive

     analytics

     in

     fields

     such

     as

     finance

     and

     healthcare

    .

     Here

     are

     some

     possible

     trends

     that

     may

     be

     seen

     in

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     accuracy

     and

     precision

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     the

     potential

     for

     more

     accurate

     and

     precise

     AI

     systems

     will

     likely

     become

     more

     widespread

    .

     This

     could

     lead

     to

     more

     effective

     predictive

     models

     and

     more

     accurate

     diagnoses

     for

     medical

     conditions

    .
    


    2

    .

     Personal

    ization

    :

     AI

     will

     continue

     to

     improve

     as

     it

     becomes

     more

     sophisticated

    ,

     and

     we

     will

     see

     a

     greater

     emphasis

     on

     personal

    izing

     the

     AI

     systems

     we

     use

    .

     This

     could

     lead

     to

     more

     efficient

     and

    



```python
llm.shutdown()
```
