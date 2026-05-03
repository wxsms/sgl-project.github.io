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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.99it/s]


    2026-05-03 23:10:23,019 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-03 23:10:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.18it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.87it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.87it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.02it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.00it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.00it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.00it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.00it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.00it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.00it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.00it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.00it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.00it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.00it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.80it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.80it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.80it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.80it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.80it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.80it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.80it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.80it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.80it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.80it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.80it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 38.81it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 38.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 26.91it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 26.91it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 26.91it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 26.91it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.28it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 24.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 24.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 24.30it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 24.30it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 24.30it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 27.71it/s]

    Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 32.80it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 32.80it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 32.80it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.80it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.80it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.80it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.59it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.59it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.59it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.59it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.59it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.59it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.14it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.14it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.14it/s]

    Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.14it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.14it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.14it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:01<00:00, 42.88it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 42.88it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 42.88it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 42.88it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 42.88it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 42.88it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.73it/s]

    Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.73it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 35.24it/s]


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
    Generated text:  Jon. I'm a human but I have the ability to communicate with animals. I help people who are in pain and distress. My name is Damp.
    If there's a way to change the weather, I can take a breath. I can also heal by giving a little or even a lot of water. I'm a different kind of animal-human. I can do these things for people.
    You've mentioned that you can communicate with animals. Could you please tell me more about your abilities?
    As a human-human, I can communicate with humans and communicate with animals. I am able to understand and empathize with the pain and suffering
    ===============================
    Prompt: The president of the United States is
    Generated text:  the leader of the Executive Branch. Their function is to execute the laws and policies of the United States government. In their capacity as the leader of the Executive Branch, the president is responsible for negotiating treaties with other countries. The president is also responsible for representing the nation in the courts of law. In addition to these duties, the president is also responsible for being involved with the advancement of American culture, literature, and the arts.
    What does the president do?
    The president of the United States is responsible for negotiating treaties with other countries and representing the United States in the courts of law. The president also represents the United States in the advancement of
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    [1] Paris  
    [2] Rome  
    [3] Berlin  
    [4] London
    
    To determine the capital of France, let's analyze the information step by step.
    
    1. Identify the question: We need to find the capital of France.
    2. Recall the list of capitals: The capitals of the European Union (EU) are listed below. However, the EU does not include the capital of France.
    3. Check the information provided: The list provided includes: London, Paris, Brussels, Dublin, and Bruges.
    4. Identify the capital: The capital of France is Paris.
    
    Therefore, the capital of France
    ===============================
    Prompt: The future of AI is
    Generated text:  predicted to be one of remarkable changes, both positive and negative. Here are some of the ways in which AI is expected to transform and shape our lives in the coming years.
    Positive impacts:
    1. Improved healthcare: AI can help doctors make more accurate diagnoses, personalize treatment plans, and improve patient outcomes.
    2. Enhanced personalization: AI can help to predict patient preferences, recommend personalized treatments, and optimize healthcare resources.
    3. Reduced healthcare costs: AI can help to automate repetitive tasks, reduce errors, and improve efficiency, ultimately leading to cost savings.
    4. Better decision-making: AI can assist with decision-making processes, from insurance claims


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What are some of your favorite hobbies or interests? I'm always looking for new experiences and opportunities to expand my knowledge and skills. What's your favorite book or movie? I'm always looking for new ways to learn and grow, and I'm always excited
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its cuisine, including its famous croissants and its traditional French wine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. Its population is over 2 million people, making it the most populous city in Europe. Paris is a city of art, culture,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration with other technologies: AI is already being integrated into a wide range of technologies, including smart homes, self-driving cars, and virtual assistants. As more and more technologies become integrated with AI, we can expect to see even more integration in the future.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our lives, there will be a greater emphasis on ethical considerations. This will include issues such as
    


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
    Generated text:  [Name], and I'm a [Role/Job/Status] at [Company/Place]. I love [Favorite Activity/Interest/Job/Question]. How can I help you today? I look forward to talking to you soon. What is your name? I'm [Name], a [Role/Job/Status] at [Company/Place]. I'm excited to meet you and learn more about you! What is your name? Hello, my name is [Name], a [Role/Job/Status] at [Company/Place]. I'm excited to meet you and learn more about you! What is your name?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its beautiful architecture, cultural richness, and vibrant nightlife. Paris is home to the Louvre Museum, Eiffel Tower, and many other iconic landmarks, as well as a rich history and world-renowned cuisine. The city also hosts numerous cultural events and festivals throughout the year, making it a popular tourist destination for visitors from all over the world. Paris is a bustling hub of artistic and intellectual life, and its reputation as one of the most beautiful cities in the world is well-earned. 
    
    Paris, the Paris, capital of France, is a fascinating city that offers a glimpse into the history and culture of this
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and diverse, with many trends developing as the technology advances. Here are some of the key trends:
    
    1. Increased Depth and Complexity: As AI becomes more powerful, it will be able to handle more complex and nuanced tasks, such as image recognition, natural language processing, and decision-making. This will allow AI to improve in a variety of ways, from better healthcare to more effective financial analysis.
    
    2. Better Explainability: As AI becomes more sophisticated, it will become easier for humans to understand how it makes decisions. This will improve the transparency of AI systems, which will be particularly important in areas like healthcare and finance.
    
    3.


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

    job

     title

    ]

     with

     [

    job

     title

    ]

     experience

    .

     I

     have

     always

     been

     passionate

     about

     [

    mention

     a

     hobby

     or

     interest

     that

     is

     relevant

     to

     the

     character

    's

     profession

     or

     background

    ],

     and

     have

     a

     deep

     appreciation

     for

     the

     work

     ethic

     and

     commitment

     required

     in

     my

     field

    .

     Whether

     it

    's

     the

     hard

     work

     that

     goes

     into

     [

    mention

     a

     specific

     aspect

     of

     my

     job

     or

     role

    ],

     or

     the

     compassion

     and

     empathy

     that

     my

     colleagues

     bring

     to

     our

     teams

    ,

     I

    'm

     here

     to

     strive

     for

     excellence

     in

     all

     that

     I

     do

    .

     What

     are

     some

     of

     the

     qualities

     that

     make

     you

     stand

     out

     from

     other

     candidates

     in

     this

     role

    ,

     and

     why

     do

     you

     believe

     these

     qualities

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     bustling

     city

     known

     for

     its

     historic

     architecture

    ,

     vibrant

     culture

    ,

     and

     French

     cuisine

    .

     Paris

     is

     also

     a

     major

     center

     of

     education

     and

     art

    .

     Additionally

    ,

     it

     is

     a

     popular

     tourist

     destination

    ,

     famous

     for

     its

     fashion

     industry

    ,

     museums

    ,

     and

     festivities

     throughout

     the

     year

    .

     Paris

     has

     a

     rich

     history

     and

     culture

     that

     has

     influenced

     much

     of

     Europe

     and

     beyond

    .

     It

     is

     a

     city

     that

     is

     steep

    ed

     in

     tradition

     and

     continues

     to

     thrive

     as

     one

     of

     the

     world

    ’s

     most

     important

     cities

    .

     
    


    Paris

    ,

     officially

     known

     as

     the

     Î

    le

    -de

    -F

    rance

    ,

     is

     located

     in

     the

     I

    le

    -de

    -F

    rance

     region

     in

     the

     north

    -central

     part

     of

     France

    .

     It

     is

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     shaped

     by

     a

     wide

     range

     of

     factors

    ,

     including

     the

     development

     of

     new

     technologies

    ,

     advances

     in

     data

     analysis

    ,

     and

     shifts

     in

     how

     AI

     is

     used

     in

     various

     industries

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

     efficiency

     and

     productivity

    :

     AI

     is

     becoming

     increasingly

     effective

     at

     autom

    ating

     repetitive

     tasks

    ,

     reducing

     costs

    ,

     and

     improving

     efficiency

     across

     a

     wide

     range

     of

     industries

    .
    


    2

    .

     Personal

    ization

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     is

     becoming

     possible

     to

     build

     personalized

     experiences

     for

     users

    ,

     based

     on

     their

     preferences

    ,

     behavior

    ,

     and

     other

     data

    .
    


    3

    .

     Enhanced

     creativity

     and

     innovation

    :

     AI

     is

     being

     used

     to

     create

     new

     forms

     of

     entertainment

    ,

     art

    ,

     and

    



```python
llm.shutdown()
```
