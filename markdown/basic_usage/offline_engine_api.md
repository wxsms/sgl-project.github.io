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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.45it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.26it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.26it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.26it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.26it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  9.26it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  9.26it/s]

    Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  9.26it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:02, 13.15it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:02, 13.15it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:04<00:02, 13.15it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:02, 13.15it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:04<00:02, 13.15it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:04<00:02, 13.15it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:04<00:02, 13.15it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:04<00:02, 13.15it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:04<00:02, 13.15it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:04<00:01, 19.74it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:04<00:01, 19.74it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:04<00:01, 19.74it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:04<00:01, 19.74it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:04<00:01, 19.74it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:04<00:01, 19.74it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:04<00:01, 19.74it/s]

    Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:04<00:01, 19.74it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:04<00:01, 19.74it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 27.12it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 27.12it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 27.12it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 27.12it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 27.12it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 27.12it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 27.12it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 27.12it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 27.12it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 34.79it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 34.79it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 34.79it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 34.79it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 34.79it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 34.79it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 34.79it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 34.79it/s] 

    Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 34.79it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.53it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.25it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.51it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.51it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 35.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 35.59it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  31%|███       | 18/58 [00:00<00:01, 35.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  31%|███       | 18/58 [00:00<00:01, 35.59it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  31%|███       | 18/58 [00:00<00:01, 35.59it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  31%|███       | 18/58 [00:00<00:01, 35.59it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.74it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.74it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.74it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.74it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.74it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.74it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.51it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.51it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.51it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.51it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.51it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.51it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  57%|█████▋    | 33/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=320 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.29it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.29it/s]

    Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.29it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.20it/s]Capturing num tokens (num_tokens=224 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.20it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.20it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.20it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.20it/s]Capturing num tokens (num_tokens=160 avail_mem=75.98 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.20it/s]Capturing num tokens (num_tokens=160 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.97it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.97it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.97it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.97it/s]

    Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.97it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.97it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=64 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.66it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.66it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.66it/s]

    Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.66it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.66it/s] Capturing num tokens (num_tokens=4 avail_mem=75.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.66it/s]Capturing num tokens (num_tokens=4 avail_mem=75.93 GB): 100%|██████████| 58/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=4 avail_mem=75.93 GB): 100%|██████████| 58/58 [00:01<00:00, 36.87it/s]


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
    Generated text:  Jesse. My sister and I are twins and we both like to draw and color. We are very good friends and we like to share our drawings with other friends. We live in a small town where no one else is like us. We are very lonely, because we have no friends and no one to share our drawings with. 32. It seems to me that you have a great idea for a cartoon. Your idea is very simple and really interesting. My friend Steve is very interested in it. What he wants to do is to make a cartoon like this. He wants to make a cartoon that's completely different from your picture
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to spend $100 billion to combat climate change or $200 billion to cut healthcare costs. The president is faced with a choice between these two actions and has limited time to make his decision. The president must spend the money in a single day, and he has a choice of two options for the day: Option 1: Spend the money to combat climate change; or Option 2: Spend the money to cut healthcare costs.
    
    Option 1: The president can spend the money on anything he wants, as long as it does not cause harm to the environment. However, the president is concerned that
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ].____
    A. Paris
    B. Brussels
    C. Munich
    D. Lille
    Answer:
    
    A
    
    The main reason for the decrease in average height of children in the Northern Hemisphere is ____
    A. Decrease in solar radiation intensity
    B. Increase in solar radiation intensity
    C. Increase in latitude
    D. Increase in altitude
    Answer:
    
    C
    
    [Multiple Choice Question] Which of the following statements about the development of land use and land transfer is correct? 
    A. The total land area of the world is about 3.6 billion hectares. 
    B. The total land area of the world is
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be very complicated and complex, but one of the biggest things that are happening is the shift towards virtual assistants. These can be anything from chatbots to virtual assistants that can help with all sorts of tasks such as answering questions, managing a calendar, and managing files and so forth. When it comes to virtual assistants, there are many different types, but the one thing that is common is that they have a learning component built into them. This means that they can continuously learn from the data that they are given to help them with their tasks. 
    
    One type of virtual assistant is a chatbot, which is a program that is designed


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I'm always looking for ways to improve my skills and knowledge. I'm also a [job title] at [company name], and I'm always eager to learn and grow. I'm a [job title] at [company name], and I'm always looking for ways to improve my skills and knowledge. I'm a [job title] at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major center for art, music, and literature. Paris is a popular tourist destination and a cultural hub for the world. It is the capital of France and the largest city in the country. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiff
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and preferences.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more widespread
    


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
    Generated text:  [Your Name], a [Your Profession/Role] with [Your Degree] in [Your Area of Expertise]. I'm passionate about [Your Passion/Favorite Subject], and I enjoy exploring the world through different perspectives. I love learning new things and finding out why they are important to others. Whether it's cooking, cooking, baking, or just exploring the world, I'm always on the lookout for new experiences and ideas. I love taking risks and challenging myself to learn and grow as a person. If you're interested in learning more about me, I'm always happy to share my thoughts and experiences with you. How about you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historic city with a rich history and is known for its landmarks such as the Eiffel Tower and the Louvre Museum. The city is also famous for its culture, cuisine, and fashion. Paris is a cosmopolitan city with a diverse population, and it is home to many of the world’s most renowned artists, writers, and performers. The French capital is a major economic and political center, and it plays a key role in the French-speaking world. It is home to the headquarters of many French-owned companies and the seat of the French government. Paris has a long history dating back to the ancient Roman Empire
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly changing with new innovations and technologies emerging constantly. Here are some possible trends that are likely to shape the future of AI:
    
    1. Enhanced intelligence: AI is getting better at recognizing patterns and making decisions, and experts predict that this will continue to improve as AI algorithms are trained to better match human intelligence.
    
    2. Emotional intelligence: AI is likely to become more empathetic and understanding, with the ability to recognize and respond to the emotions of its users. This will enable AI to better understand human behavior and interactions.
    
    3. Quantum computing: Quantum computers are expected to become even more powerful and faster than traditional computers, which could lead to breakthrough


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

    What

     is

     your

     character

    's

     profession

    ,

     background

    ,

     or

     role

     in

     the

     story

    ]?

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     about

     you

    .

     Let

    's

     get

     to

     know

     each

     other

    !

     What

     is

     your

     name

    ,

     and

     what

     is

     your

     profession

     or

     background

    ?

     How

     do

     you

     typically

     communicate

     with

     your

     audience

    ?

     Do

     you

     have

     any

     particular

     interests

     or

     hobbies

     that

     I

     should

     be

     aware

     of

    ?

     Let

     me

     know

    ,

     and

     I

    'll

     get

     started

     on

     my

     self

    -int

    roduction

    !

     [

    Your

     Name

    ]

     [

    Your

     Profession

    /

    Background

    /

    Role

    ]

     [

    Your

     Inter

    ests

    /H

    obbies

    ]

     [

    Your

     Name

    ]

     [

    Your

     Profession

    /

    Background

    /

    Role

    ]

     [

    Your

     Inter

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Hay

    e

    ."
    


    This

     statement

     captures

     the

     core

     fact

     about

     Paris

    ,

     including

     its

     status

     as

     the

     capital

     of

     France

     and

     the

     city

    's

     name

     in

     French

    ,

     which

     is

     why

     it

     is

     commonly

     referred

     to

     as

     "

    La

     Hay

    e

    ."

     It

     provides

     a

     brief

    ,

     informative

     overview

     of

     the

     capital

     city

    's

     importance

     and

     cultural

     significance

     in

     France

    .

     Additional

     details

     could

     include

     the

     city

    's

     historical

     origins

    ,

     population

    ,

     or

     notable

     landmarks

    ,

     depending

     on

     the

     context

    .

     The

     statement

     is

     concise

     yet

     comprehensive

    ,

     giving

     readers

     an

     overview

     of

     Paris

    's

     significance

    .

     A

     more

     detailed

     exploration

     of

     Paris

    's

     history

    ,

     culture

    ,

     and

     current

     affairs

     might

     expand

     on

     these

     points

     in

     a

     broader

     context

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     very

     complex

     and

     diverse

    ,

     with

     many

     different

     areas

     of

     development

     and

     innovation

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

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

     is

     already

     being

     used

     to

     improve

     patient

     care

    ,

     from

     analyzing

     medical

     images

     to

     predicting

     disease

     progression

    .

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     even

     more

     advanced

     healthcare

     applications

    .
    


    2

    .

     Enhanced

     AI

     for

     autonomous

     vehicles

    :

     As

     autonomous

     vehicle

     technology

     becomes

     more

     widely

     used

    ,

     AI

     will

     play

     an

     even

     bigger

     role

     in

     transportation

    .

     Autonomous

     vehicles

     will

     be

     able

     to

     make

     decisions

     based

     on

     a

     variety

     of

     sensors

     and

     algorithms

    ,

     making

     them

     more

     efficient

     and

     safer

    .
    


    3

    .

     Personal

    ized

     AI

    



```python
llm.shutdown()
```
