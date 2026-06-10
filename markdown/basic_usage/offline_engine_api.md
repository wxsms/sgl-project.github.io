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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:07,  6.16it/s]

    Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:04<00:07,  6.16it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 11.75it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 11.75it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 11.75it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 11.75it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 11.75it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 11.75it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 11.75it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 16.41it/s]

    Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:01, 16.41it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 25.41it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 25.41it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 25.41it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 25.41it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 25.41it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 25.41it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:05<00:00, 25.41it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]

    Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 36.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.42it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.42it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.42it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.52it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.93it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.93it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 37.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 37.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 37.40it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  31%|███       | 18/58 [00:00<00:01, 37.40it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 37.40it/s] Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=896 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=704 avail_mem=74.09 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.27it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.27it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.27it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.27it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.27it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:00<00:00, 35.10it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:00<00:00, 35.10it/s]Capturing num tokens (num_tokens=384 avail_mem=74.08 GB):  53%|█████▎    | 31/58 [00:00<00:00, 35.10it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  53%|█████▎    | 31/58 [00:00<00:00, 35.10it/s]Capturing num tokens (num_tokens=320 avail_mem=74.07 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.10it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.10it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.93it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.93it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.93it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.93it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.93it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.93it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=176 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.04 GB):  71%|███████   | 41/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=112 avail_mem=74.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.41it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.41it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.41it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.41it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.41it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.41it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.36it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.04it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.04it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 36.91it/s]


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
    Generated text:  Nick, and I have recently completed a course in Python programming. As of now, I am learning to use Python for data analysis and visualization. Could you please provide some examples of how to use Python for data analysis and visualization? Additionally, I would like to ask about a specific data set that I have and want to explore its characteristics. Please provide some Python code to assist you in understanding the data better. Finally, I would like to ask about the latest trends and technologies in the field of data science, as well as some guidance on how to apply these in my daily work. Lastly, please share any tips or tricks that you think
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. The president works for the country, and it’s very important for the country to have the president. There are two main things that the president does. First, the president speaks to the country. The president speaks to the people of the country and talks about what the president thinks. This is important because the people of the country need to know what the president thinks. The second thing the president does is to make decisions. The president does this by signing the laws into being. For example, if there is a problem with a person getting a job, the president can make a law that says, "If a person
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It's a bustling city of about 2.3 million people. It was built on the banks of the Seine River. The city's most famous landmark is the Eiffel Tower, a bronze tower that stands 324 feet tall. It was designed by a French architect, Gustave Eiffel. 
    
    Imagine a magical island where you can walk through the Eiffel Tower. You start at the bottom of the tower and walk straight up. After walking through the tower, you decide to go around the island and come back to the ground. To make the trip as fun as walking through the Eiff
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable. The field is changing so rapidly that some people suggest the world is about to go into a mental break. The reasons behind this are many, ranging from new technologies to the security of artificial intelligence. I believe that the future of AI is safe, and that it will be a critical part of the transformation of our society and economies. But to know for sure, we need to research this topic thoroughly. There are multiple factors that impact the development of AI, and if we continue to ignore them, we may end up in the same situation that caused the major problems in the past.
    
    AI is a rapidly growing field, and as such


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm passionate about [reason for being at the company]. I'm always looking for ways to [something specific, like learning new skills, improving my work ethic, etc.]. I'm a [type of person, like introverted, extroverted, etc.]. I enjoy [something enjoyable, like hiking, cooking, etc.]. I'm [any other personal traits, like friendly, outgoing, etc.]. I'm excited to [what you'd like to do next, like start a new project, attend a meeting, etc.]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many important museums and historical sites. Paris is a popular tourist destination and a major economic center, with a rich cultural heritage and a vibrant nightlife. The city is also known for its diverse cuisine and its role in the French language and literature. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also home to many international organizations and institutions, including the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be an increased focus on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. Development of more advanced AI systems: As AI technology continues to advance, there will be an increased focus on developing more advanced AI systems that can perform tasks that were previously considered impossible or dangerous.
    
    3. Integration of AI with other technologies: AI is already being
    


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
    Generated text:  [Your Name], and I'm [Age]. I'm a [职业] with over [number] years of experience in the [industry] field. I enjoy [your passion or hobby] and have always been passionate about [your area of interest or personality trait]. I'm always striving to learn and grow in my field, always looking for ways to improve myself and become even better than I am. My goal is to make a positive impact on the world and contribute to society in a meaningful way. I am [interest or personality trait] and I enjoy [something that excites you]. I'm also [something else that excites
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Here's a concise fact:
    
    Paris is the largest city in France and its capital.
    
    Paris is known for its iconic landmarks such as the Eiffel Tower, Notre Dame Cathedral, Louvre Museum, and the Louvre Gallery. It's also home to the Louvre Museum and the Musée d'Orsay. Paris is a historic city with a rich history dating back to ancient times. The French Revolution, the French Revolution, and the French Revolution were all centered in Paris. As a major transportation hub and gateway to Europe, Paris is a bustling urban center with many attractions for both locals and tourists. Paris is home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities and potential challenges, and there is no telling what the future will hold. However, here are some possible future trends in AI that are currently being explored and discussed:
    
    1. Increased use of AI in healthcare: One of the most promising areas for AI in the future is in healthcare. AI can help doctors and researchers to analyze medical images, predict patient outcomes, and develop new treatments. AI-powered virtual assistants, such as chatbots and virtual assistants, may also become more common in medical settings to help patients and healthcare providers.
    
    2. Development of AI-driven personalized medicine: AI can help healthcare providers to tailor treatment plans to


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

    occupation

    ].

     I

    'm

     always

     eager

     to

     learn

    ,

     so

     I

     don

    't

     mind

     spending

     time

     with

     people

     of

     all

     ages

     and

     backgrounds

    .

     I

    'm

     also

     a

     [

    skill

     or

     talent

    ]

     in

     my

     field

    ,

     and

     I

     enjoy

     sharing

     my

     knowledge

     with

     others

    .


    As

     someone

     who

     values

     honesty

     and

     integrity

    ,

     I

     strive

     to

     always

     act

     with

     integrity

     and

     show

     respect

     to

     all

     people

    .

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     myself

     and

     continue

     to

     learn

     new

     things

    .


    If

     you

    're

     interested

     in

     learning

     more

     about

     me

    ,

     please

     let

     me

     know

     and

     I

    'll

     be

     happy

     to

     answer

     any

     questions

     you

     may

     have

    .

     Looking

     forward

     to

     talking

     to

     you

    !

     #

    self

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     

    6

    th

     largest

     city

     in

     the

     world

     by

     population

     and

     one

     of

     the

     most

     visited

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

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Op

    éra

     Garn

    ier

    .

     Paris

     is

     a

     cultural

    ,

     historical

    ,

     and

     political

     hub

    ,

     and

     the

     city

     is

     home

     to

     many

     world

    -ren

    owned

     institutions

     and

     attractions

    .

     It

     is

     also

     known

     for

     its

     rich

     history

     and

     artistic

     tradition

    ,

     which

     can

     be

     seen

     in

     the

     city

    's

     architecture

    ,

     museums

    ,

     and

     food

     scene

    .

     Paris

     is

     a

     city

     of

     contrasts

     and

     beauty

    ,

     with

     its

     urban

     landscapes

    ,

     charming

     cafes

    ,

     and

     lively

     nightlife

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     increasingly

     complex

    ,

     with

     many

     possible

     trends

     shaping

     the

     technology

    's

     direction

    .

     Here

     are

     some

     potential

     areas

     of

     focus

     in

     AI

    :
    


    1

    .

     AI

     ethics

     and

     governance

    :

     AI

     will

     continue

     to

     evolve

     and

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     but

     we

     will

     need

     to

     develop

     guidelines

     and

     regulations

     to

     ensure

     its

     ethical

     use

    .

     This

     includes

     considerations

     of

     bias

    ,

     privacy

    ,

     and

     transparency

    .
    


    2

    .

     AI

    -powered

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     diagnose

     and

     treat

     diseases

    ,

     but

     more

     will

     be

     developed

     to

     predict

     and

     prevent

     illnesses

    .

     AI

    -powered

     healthcare

     will

     likely

     become

     more

     widely

     available

     and

     affordable

     in

     the

     future

    ,

     enabling

     more

     widespread

     access

     to

     healthcare

    .
    


    3

    .

     AI

    -powered

     education

    :

    



```python
llm.shutdown()
```
