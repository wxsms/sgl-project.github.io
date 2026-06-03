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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.31it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:06,  1.22s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:06,  1.22s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:06,  1.22s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]

    Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  6.53it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  6.53it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  6.53it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  6.53it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  6.53it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:05<00:06,  6.53it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:05<00:06,  6.53it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:03, 11.47it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:03, 11.47it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:03, 11.47it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:03, 11.47it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:03, 11.47it/s]

    Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:03, 11.47it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:03, 11.47it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:01, 17.10it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:01, 17.10it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:01, 17.10it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:01, 17.10it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:01, 17.10it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:05<00:01, 17.10it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:05<00:01, 17.10it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 22.87it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 22.87it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 22.87it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 22.87it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 22.87it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 22.87it/s]

    Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 22.87it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 22.87it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 30.43it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 37.61it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 37.61it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 37.61it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 37.61it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 37.61it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 37.61it/s]

    Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 37.61it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 37.61it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 44.21it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 44.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.53 GB):   3%|▎         | 2/58 [00:00<00:05, 11.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.72 GB):   3%|▎         | 2/58 [00:00<00:05, 11.00it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=54.72 GB):   3%|▎         | 2/58 [00:00<00:05, 11.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=54.72 GB):   7%|▋         | 4/58 [00:00<00:04, 13.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.71 GB):   7%|▋         | 4/58 [00:00<00:04, 13.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.70 GB):   7%|▋         | 4/58 [00:00<00:04, 13.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.70 GB):  10%|█         | 6/58 [00:00<00:03, 15.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.69 GB):  10%|█         | 6/58 [00:00<00:03, 15.14it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=54.68 GB):  10%|█         | 6/58 [00:00<00:03, 15.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.68 GB):  10%|█         | 6/58 [00:00<00:03, 15.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.68 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.57 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.66 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.66 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.04it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=54.66 GB):  21%|██        | 12/58 [00:00<00:03, 12.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.65 GB):  21%|██        | 12/58 [00:00<00:03, 12.60it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.65 GB):  21%|██        | 12/58 [00:00<00:03, 12.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.62 GB):  21%|██        | 12/58 [00:00<00:03, 12.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.60 GB):  21%|██        | 12/58 [00:00<00:03, 12.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.60 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.06it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.61 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.62 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.06it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.61 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.06it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=54.61 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.61 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.45it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.59 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.45it/s]Capturing num tokens (num_tokens=960 avail_mem=54.58 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.45it/s] Capturing num tokens (num_tokens=896 avail_mem=54.57 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.45it/s]Capturing num tokens (num_tokens=896 avail_mem=54.57 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.25it/s]Capturing num tokens (num_tokens=832 avail_mem=54.57 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.25it/s]Capturing num tokens (num_tokens=768 avail_mem=54.57 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.25it/s]

    Capturing num tokens (num_tokens=704 avail_mem=54.56 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.25it/s]Capturing num tokens (num_tokens=640 avail_mem=54.58 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.25it/s]Capturing num tokens (num_tokens=640 avail_mem=54.58 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.95it/s]Capturing num tokens (num_tokens=576 avail_mem=54.57 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.95it/s]Capturing num tokens (num_tokens=512 avail_mem=54.55 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.95it/s]Capturing num tokens (num_tokens=480 avail_mem=54.54 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.95it/s]Capturing num tokens (num_tokens=448 avail_mem=54.56 GB):  47%|████▋     | 27/58 [00:01<00:01, 25.95it/s]Capturing num tokens (num_tokens=448 avail_mem=54.56 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.96it/s]Capturing num tokens (num_tokens=416 avail_mem=54.56 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.96it/s]Capturing num tokens (num_tokens=384 avail_mem=54.55 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.96it/s]

    Capturing num tokens (num_tokens=352 avail_mem=54.54 GB):  53%|█████▎    | 31/58 [00:01<00:00, 27.96it/s]Capturing num tokens (num_tokens=352 avail_mem=54.54 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.92it/s]Capturing num tokens (num_tokens=320 avail_mem=54.53 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.92it/s]Capturing num tokens (num_tokens=288 avail_mem=54.52 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.92it/s]Capturing num tokens (num_tokens=256 avail_mem=54.52 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.92it/s]Capturing num tokens (num_tokens=240 avail_mem=54.51 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.92it/s]Capturing num tokens (num_tokens=240 avail_mem=54.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.22it/s]Capturing num tokens (num_tokens=224 avail_mem=54.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.22it/s]

    Capturing num tokens (num_tokens=208 avail_mem=54.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.22it/s]Capturing num tokens (num_tokens=192 avail_mem=54.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.22it/s]Capturing num tokens (num_tokens=176 avail_mem=54.47 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.22it/s]Capturing num tokens (num_tokens=176 avail_mem=54.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.29it/s]Capturing num tokens (num_tokens=160 avail_mem=54.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.29it/s]Capturing num tokens (num_tokens=144 avail_mem=54.46 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.29it/s]

    Capturing num tokens (num_tokens=128 avail_mem=54.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.29it/s]Capturing num tokens (num_tokens=128 avail_mem=54.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 26.73it/s]Capturing num tokens (num_tokens=112 avail_mem=54.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 26.73it/s]Capturing num tokens (num_tokens=96 avail_mem=54.46 GB):  78%|███████▊  | 45/58 [00:02<00:00, 26.73it/s] Capturing num tokens (num_tokens=80 avail_mem=54.45 GB):  78%|███████▊  | 45/58 [00:02<00:00, 26.73it/s]Capturing num tokens (num_tokens=80 avail_mem=54.45 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.46it/s]Capturing num tokens (num_tokens=64 avail_mem=54.45 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.46it/s]

    Capturing num tokens (num_tokens=48 avail_mem=54.43 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.46it/s]Capturing num tokens (num_tokens=32 avail_mem=54.43 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.46it/s]Capturing num tokens (num_tokens=32 avail_mem=54.43 GB):  88%|████████▊ | 51/58 [00:02<00:00, 22.28it/s]Capturing num tokens (num_tokens=28 avail_mem=54.42 GB):  88%|████████▊ | 51/58 [00:02<00:00, 22.28it/s]Capturing num tokens (num_tokens=24 avail_mem=54.42 GB):  88%|████████▊ | 51/58 [00:02<00:00, 22.28it/s]

    Capturing num tokens (num_tokens=20 avail_mem=54.41 GB):  88%|████████▊ | 51/58 [00:02<00:00, 22.28it/s]Capturing num tokens (num_tokens=20 avail_mem=54.41 GB):  93%|█████████▎| 54/58 [00:02<00:00, 23.31it/s]Capturing num tokens (num_tokens=16 avail_mem=54.41 GB):  93%|█████████▎| 54/58 [00:02<00:00, 23.31it/s]Capturing num tokens (num_tokens=12 avail_mem=54.41 GB):  93%|█████████▎| 54/58 [00:02<00:00, 23.31it/s]Capturing num tokens (num_tokens=8 avail_mem=54.40 GB):  93%|█████████▎| 54/58 [00:02<00:00, 23.31it/s] Capturing num tokens (num_tokens=4 avail_mem=54.40 GB):  93%|█████████▎| 54/58 [00:02<00:00, 23.31it/s]Capturing num tokens (num_tokens=4 avail_mem=54.40 GB): 100%|██████████| 58/58 [00:02<00:00, 23.26it/s]


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
    Generated text:  Tanya, I am 16 years old, I live in the USA. I want to ask a question about the United States. Which one is the best way to approach it?
    
    1) I do not know the answer to a question that asks for information about the United States. I do not want to ask someone for it.
    
    2) I do not want to ask anyone for information about the United States because it is too much work for me.
    
    3) I want to ask a question about the United States, and I want to do it in my own way, that is, I want to know about the United States.
    
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposed by the Speaker of the House and the President of the Senate. With the president's approval, the Senate confirms the executive branch's nomination of a person to fill a position. There is no power to declare the president as unfit to hold office or to impeach the president.
    
    I am trying to convince my boss that this is a bad idea and want to make sure I understand everything correctly.
    
    **My thought:** 
    
    1. The president is appointed by the Speaker of the House and the President of the Senate. This creates a check and balance between the executive and legislative branches.
    2. A presidential candidate can run for president and be nominated
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 正确
    B. 错误
    Answer:
    B
    
    The smallest frame of the 1989 French National Economy Report is 32 cm.
    A. 正确
    B. 错误
    Answer:
    B
    
    The largest frame of the 1989 French National Economy Report is 37 cm.
    A. 正确
    B. 错误
    Answer:
    A
    
    In the financial system of a certain country, the credit intermediation ratio is 50%. What does this mean?
    A. 50% of the total financial assets should be transferred
    ===============================
    Prompt: The future of AI is
    Generated text:  coming fast
    
    Artificial intelligence has already started changing the way that we live and work. It’s going to continue to revolutionize our world in the coming years.
    
    Artificial intelligence will be used to improve the lives of people all around the world. It will be used to make healthcare more efficient, to improve transportation, to help manage data, and more. The future of AI is going to be exciting, but we have to be careful.
    
    There is a growing concern that AI will become too powerful and will cause harm to people. However, there are also benefits to AI, and these benefits are being realized.
    
    Artificial intelligence is used


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of French culture and its role in the French Revolution and the French Revolution. It is also a popular tourist destination, attracting millions of visitors each year. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has played a significant role in shaping French culture and identity. The city is also home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation and efficiency: AI is expected to continue to automate a wide range of tasks, from manufacturing to customer service. This will lead to increased efficiency and productivity, as machines can perform tasks that would previously require human intervention.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need for privacy and security. AI systems will need to be designed with privacy and security in mind, and there will be a push for regulations and standards to ensure that AI is used responsibly.
    
    3. Greater focus on ethical AI: As AI becomes more
    


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
    Generated text:  [Your Name], and I am a [Your Profession] with a passion for [Your Passion]. I'm a [Your Age], [Your Height], and [Your Weight] and I was born in [Your Birthplace]. I am a [Your Personality Trait], [Your Unique Selling Point], and [Your Reward]. I am [Your Specialization or Expertise], [Your Favorite Color], and [Your Favorite Book]. I'm always looking for new opportunities and always striving to grow and learn. My biggest obstacle is [Your Past or Challenges]. I have a keen understanding of [Your Industry/Field], and I'm always trying
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, an iconic city famous for its rich history, famous landmarks, and diverse culture.
    
    What are some key attractions or landmarks in Paris that tourists should visit? Some key attractions and landmarks in Paris include the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, Montmartre, Sacré-Cœur Basilica, and the Champs-Élysées. These landmarks and attractions are a must-visit for anyone looking to explore and enjoy Paris. 
    
    Based on the text provided, what are some key attractions or landmarks in Paris that tourists should consider visiting?
    
    Some key attractions or landmarks in Paris that tourists should consider visiting include
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by many different trends that will shape the technology's progress, development and adoption in various industries. Here are some possible trends in AI that are currently being researched and developed:
    
    1. Deep Learning: Deep learning is a form of AI that uses neural networks to learn from vast amounts of data. Deep learning is expected to have a significant impact on a variety of applications, including natural language processing, computer vision, and speech recognition.
    
    2. Explainable AI: The problem of transparency and interpretability of AI systems has been a significant challenge in the field of AI. Explanable AI is a promising area where researchers are exploring


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

    Type

    ]

     who

     has

     a

     passion

     for

     [

    Your

     passion

     or

     hobby

    ].

     I

    'm

     a

     [

    Age

    ]

     year

     old

     and

     I

    've

     always

     been

     passionate

     about

     [

    Your

     passion

     or

     hobby

    ].

     I

    'm

     someone

     who

     thr

    ives

     on

     learning

     new

     things

     and

     challenging

     myself

     in

     my

     daily

     life

    .

     I

     believe

     in

     the

     power

     of

     creativity

     and

     innovation

    ,

     and

     I

    'm

     always

     looking

     for

     new

     and

     exciting

     ways

     to

     make

     my

     life

     and

     the

     world

     around

     me

     better

    .

     I

    'm

     excited

     to

     dive

     into

     any

     new

     challenges

     and

     experiences

     that

     come

     my

     way

    .

     Thank

     you

     for

     having

     me

    !

     

    🌟

    ✨

     #

    Personal

    Reflection

    s

     #

    Creative

    Life

     #

    New

    Op

    port

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     the

     most

     populous

     city

     in

     the

     European

     Union

     and

     is

     a

     cosm

    opolitan

     city

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

    ,

     and

     its

     many

     UNESCO

     world

     heritage

     sites

     highlight

     its

     cultural

     and

     architectural

     heritage

    .

     Paris

     is

     known

     for

     its

     fashion

    ,

     food

    ,

     and

     wine

    ,

     and

     it

     is

     home

     to

     many

     world

    -ren

    owned

     museums

    ,

     such

     as

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     The

     city

     has

     a

     diverse

     population

     and

     is

     a

     significant

     economic

     and

     cultural

     hub

     in

     the

     country

    .

     Paris

     also

     hosts

     major

     events

     such

     as

     the

     Olympics

     and

     the

     E

    iff

    el

     Tower

    .

     As

     of

     

    2

    0

    2

    1

    ,

     Paris

     has

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     potential

     and

     exciting

     possibilities

    ,

     but

     it

    's

     important

     to

     note

     that

     AI

     is

     still

     in

     its

     early

     stages

     and

     there

     are

     many

     challenges

     and

     unknown

    s

     to

     consider

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     in

     the

     next

     few

     years

    :
    


    1

    .

     Increased

     accuracy

    :

     As

     AI

     continues

     to

     improve

    ,

     it

    's

     likely

     to

     achieve

     higher

     levels

     of

     accuracy

     and

     precision

     in

     its

     predictions

     and

     decisions

    .

     This

     could

     lead

     to

     more

     reliable

     and

     trustworthy

     AI

     systems

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     AI

     is

     already

     being

     integrated

     with

     other

     technologies

     such

     as

     sensors

    ,

     machines

    ,

     and

     blockchain

    .

     As

     these

     technologies

     evolve

    ,

     AI

     could

     become

     even

     more

     integrated

     with

     them

    ,

     leading

     to

     even

     greater

     complexity

     and

    



```python
llm.shutdown()
```
