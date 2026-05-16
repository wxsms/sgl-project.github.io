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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.39it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.38it/s]


    2026-05-16 13:06:06,284 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 13:06:06] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.45it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.96it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.97it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.11it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.11it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.11it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.11it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.11it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.11it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.11it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.11it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.11it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.11it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.11it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.05it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.05it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   2%|▏         | 1/58 [00:00<00:08,  6.54it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.76 GB):   2%|▏         | 1/58 [00:00<00:08,  6.54it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   2%|▏         | 1/58 [00:00<00:08,  6.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   5%|▌         | 3/58 [00:00<00:04, 11.59it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.26 GB):   5%|▌         | 3/58 [00:00<00:04, 11.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   5%|▌         | 3/58 [00:00<00:04, 11.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   5%|▌         | 3/58 [00:00<00:04, 11.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):  10%|█         | 6/58 [00:00<00:02, 17.84it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):  10%|█         | 6/58 [00:00<00:02, 17.84it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:02, 17.84it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:02, 17.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:02, 17.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.39it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.73it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.73it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.57it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.57it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.57it/s]

    Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.57it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.57it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.57it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.75it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.75it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.75it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.75it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.75it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  52%|█████▏    | 30/58 [00:01<00:00, 42.75it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  60%|██████    | 35/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  60%|██████    | 35/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 44.34it/s]

    Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  60%|██████    | 35/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  60%|██████    | 35/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.94it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.94it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 47.40it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.33it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.33it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.85it/s]


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
    Generated text:  __________. I'm from ______. I like ________ best. My hobby is _________. Where are you from? ______________.
    A. My name is Zhenfeng. I'm from Jiangmen. I like collecting stamps best. My hobby is drawing. I'm from Beijing.
    B. My name is Zhenfeng. I'm from Jiangmen. I like collecting stamps best. My hobby is drawing. I'm from Beijing.
    C. My name is Zhenfeng. I'm from Jiangmen. I like collecting stamps best. My hobby is drawing. I'm from Beijing.
    D. My name
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to enforce a new law that will give the government more control over the economy. The law, if passed, will force the government to spend more money than is needed to pay for the spending. The president has thought about how to vote on the bill and has asked you, the finance officer, to help with the decision-making process.
    
    The president has requested a detailed breakdown of the costs and benefits of the proposed law. You have been given the following information:
    
    - The current budget deficit is $300 billion
    - The government will spend $100 billion to pay for this spending
    - The proposed law
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Lille
    C. Marseille
    D. Dijon
    Answer:
    
    A
    
    The author of "The Red and the Black" is ____.
    A. Alain Robbe-Grillet
    B. Leo Tolstoy
    C. Maxim Gorky
    D. Eugene Ionesco
    Answer:
    
    A
    
    The characteristic of a liability to the government is ____.
    A. Debt-to-Gross Domestic Product ratio
    B. Debt-to-Gross National Product ratio
    C. Debt-to-GDP ratio
    D. Debt-to-GDP ratio of the government sector
    Answer:
    
    A
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable. While it's clear that the growth of AI will be a major driver of our economic growth, there are uncertainties. For example, AI has the potential to dramatically change the way we work and interact with each other, but it also has the potential to create new jobs and opportunities that may not be fully realized. So, while we're confident that AI has the potential to transform the world, we need to be prepared for all the challenges and opportunities that come with it.
    What are some potential job disruptions that may result from the rapid development of AI?
    Some potential job disruptions that may result from the rapid development of AI include job displacement


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [job title] at [company name], and I've been working in [industry] for [number of years] years. I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field. What do you do for a living? I'm a [job title] at [company name], and I've been working in [industry] for [number of years] years. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination and is known for its rich history, art, and cuisine. The city is also home to the French Parliament and the French Academy of Sciences. 
    
    Therefore, the statement "The capital of France is Paris" is true. However, it is important to note
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection, risk management, and investment decision-making
    


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
    Generated text:  [Name], and I'm a/an [Job title] at [Company name]. I'm excited to be here today. I've been working hard for [reason for job] at [company], and I'm always looking for ways to improve and grow as a professional. I'm a true-blue, confident person who always strives to be the best at what I do. I'm passionate about [reason for passion], and I enjoy helping others achieve their goals. Thank you for taking the time to meet me, and I look forward to continuing this journey together. [Name]... [Short bio (max 50 words)] This
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its historical significance and diverse cultural scene. It's a bustling city with iconic landmarks like the Eiffel Tower and the Notre-Dame Cathedral. It's a popular tourist destination, especially for its beautiful architecture and world-renowned cuisine. Paris has a long and storied history dating back to the 6th century, making it a fascinating city for history buffs. The city is also home to many famous art museums, including the Louvre and the Musée d'Orsay. Paris is a cultural melting pot of France and the world, with a diverse range of food and drink options, and is the third-largest
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a growing number of applications in various sectors, including healthcare, transportation, and entertainment. Some potential trends in AI include:
    
    1. Increased accuracy and precision: As AI models become more sophisticated and capable, they are expected to continue improving their accuracy and precision in predicting outcomes, detecting anomalies, and making decisions.
    
    2. Enhanced empathy: AI is already being used in fields such as healthcare and customer service to simulate human-like interactions and provide emotional support. As AI technology continues to evolve, it is likely that we will see even greater improvements in empathy and understanding.
    
    3. Integration with traditional industries: AI is already being integrated


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

     professional

     software

     developer

     with

     over

     five

     years

     of

     experience

     in

     creating

     and

     maintaining

     cutting

    -edge

     software

     solutions

    .

     I

     am

     an

     efficient

     and

     detail

    -oriented

     person

     who

     thr

    ives

     in

     a

     fast

    -paced

     environment

     and

     am

     always

     looking

     for

     ways

     to

     improve

     my

     skills

    .

     I

     am

     passionate

     about

     working

     on

     complex

     projects

     and

     have

     a

     strong

     work

     ethic

    ,

     which

     has

     allowed

     me

     to

     consistently

     exceed

     my

     targets

     and

     get

     ahead

     in

     my

     career

    .

     I

     am

     also

     someone

     who

     enjoys

     being

     creative

    ,

     collaborating

     with

     others

    ,

     and

     learning

     new

     things

    .

     What

    ’s

     your

     name

    ?

     What

    ’s

     your

     profession

    ?

     What

     kind

     of

     software

     do

     you

     specialize

     in

    ?

     What

     kind

     of

     projects

     have

     you

     worked

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     oldest

     continuously

     inhabited

     city

     in

     the

     world

    ,

     located

     in

     the

     north

    western

     part

     of

     the

     country

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     culture

    .

     Paris

     is

     home

     to

     many

     famous

     museums

    ,

     fashion

     houses

    ,

     and

     artistic

     landmarks

    .

     The

     city

     is

     also

     known

     for

     its

     annual

     Bast

    ille

     Day

     celebrations

     and

     its

     role

     in

     the

     French

     Revolution

    .

     With

     its

     iconic

     E

    iff

    el

     Tower

     and

     Se

    ine

     River

    ,

     Paris

     is

     a

     unique

     and

     enchant

    ing

     city

     that

     has

     captured

     the

     hearts

     of

     people

     all

     over

     the

     world

    .

     

     When

     it

     comes

     to

     the

     capital

     of

     France

    ,

     no

     city

     is

     more

     iconic

     and

     has

     a

     much

     deeper

     cultural

     and

     historical

     significance

     than

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     changing

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

     industry

     in

     the

     coming

     years

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

     artificial

     intelligence

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     AI

     continues to

     advance

    ,

     there

     is

     a

     growing

     concern

     about

     the

     ethical

     implications

     of

     its

     use

    .

     Governments

    ,

     companies

    ,

     and

     organizations

     are

     increasingly

     addressing

     AI

     development

     and

     deployment

     with

     a

     focus

     on

     creating

     ethical

     frameworks

     and

     regulations

     that

     govern

     its

     development

     and

     usage

    .
    


    2

    .

     Expansion

     of

     AI

     applications

    :

     As

     AI

     continues

     to

     develop

    ,

     it

     is

     expected

     to

     have

     a

     greater

     impact

     on

     various

     industries

    ,

     including

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     more

    .

     AI

     is

     expected

     to

    



```python
llm.shutdown()
```
