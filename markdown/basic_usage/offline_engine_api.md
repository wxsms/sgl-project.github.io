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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.45it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.54it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.54it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.54it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.54it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.54it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.54it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.54it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.54it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.54it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.54it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 31.61it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 31.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.87it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.87it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):   9%|▊         | 5/58 [00:00<00:02, 21.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.00 GB):   9%|▊         | 5/58 [00:00<00:02, 21.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.56it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.56it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.56it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.56it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=73.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=960 avail_mem=73.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=73.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.22it/s]Capturing num tokens (num_tokens=896 avail_mem=73.96 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.09it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.09it/s]Capturing num tokens (num_tokens=768 avail_mem=73.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.09it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.09it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.09it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.77it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.77it/s]Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.77it/s]Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.77it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.77it/s]

    Capturing num tokens (num_tokens=416 avail_mem=73.67 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.77it/s]Capturing num tokens (num_tokens=416 avail_mem=73.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=384 avail_mem=73.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=352 avail_mem=73.64 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=320 avail_mem=72.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=288 avail_mem=72.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=256 avail_mem=72.95 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.39it/s]Capturing num tokens (num_tokens=256 avail_mem=72.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=240 avail_mem=72.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=224 avail_mem=72.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=208 avail_mem=72.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=192 avail_mem=72.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.06it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=160 avail_mem=72.93 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=160 avail_mem=72.93 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.73it/s]Capturing num tokens (num_tokens=144 avail_mem=72.93 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.73it/s]Capturing num tokens (num_tokens=128 avail_mem=72.93 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.73it/s]Capturing num tokens (num_tokens=112 avail_mem=72.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.73it/s]Capturing num tokens (num_tokens=96 avail_mem=72.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.73it/s] Capturing num tokens (num_tokens=80 avail_mem=72.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.73it/s]Capturing num tokens (num_tokens=80 avail_mem=72.92 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.57it/s]Capturing num tokens (num_tokens=64 avail_mem=72.91 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.57it/s]Capturing num tokens (num_tokens=48 avail_mem=72.91 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.57it/s]Capturing num tokens (num_tokens=32 avail_mem=72.91 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.57it/s]

    Capturing num tokens (num_tokens=28 avail_mem=72.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.57it/s]Capturing num tokens (num_tokens=24 avail_mem=72.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.57it/s]Capturing num tokens (num_tokens=24 avail_mem=72.90 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.83it/s]Capturing num tokens (num_tokens=20 avail_mem=72.90 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.83it/s]Capturing num tokens (num_tokens=16 avail_mem=72.90 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.83it/s]Capturing num tokens (num_tokens=12 avail_mem=72.89 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.83it/s]Capturing num tokens (num_tokens=8 avail_mem=72.89 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.83it/s] Capturing num tokens (num_tokens=4 avail_mem=72.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.83it/s]Capturing num tokens (num_tokens=4 avail_mem=72.88 GB): 100%|██████████| 58/58 [00:01<00:00, 46.49it/s]Capturing num tokens (num_tokens=4 avail_mem=72.88 GB): 100%|██████████| 58/58 [00:01<00:00, 39.59it/s]


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
    Generated text:  Susan. I am from England. I often go to school by car. I have many friends in China. They are all very nice to me. Sometimes they say sorry to me. I think it's not polite to say sorry to my friends in China. But my parents often say to me "I love you", which makes me feel very happy. I know it is important to say sorry. I also know it is not polite to say sorry to my friends in China. I tell my friends that I love them and I will try to be better. This is my first time to say sorry. It's a good way to tell
    ===============================
    Prompt: The president of the United States is
    Generated text:  a male. The current president of the United States is a citizen of Canada. The only other possibility is that the president of the United States is a member of the Chinese government.
    Which of the following statements is supported by the argument given?
    A) The president of the United States is a member of the Chinese government.
    B) The president of the United States is a citizen of Canada.
    C) The president of the United States is a member of the Chinese government.
    D) The president of the United States is a citizen of Canada.
    E) The president of the United States is a member of the Chinese government.
    
    The logical reasoning involved
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Paris is a major cultural and economic center in the northeastern part of France. It is the largest city in France, with an estimated population of over 2.1 million people. The city lies on the banks of the river Seine, which is the longest river in France and the third longest river in the world after the Nile and the Amazon. The Seine has been a natural waterway since the Neolithic age, and it is still used for transportation today.
    
    The city of Paris has a long and rich history, dating back to the Roman Empire. The city was an important trading center during the Middle Ages, and it
    ===============================
    Prompt: The future of AI is
    Generated text:  here, but what is AI? It's all about the ability to automate decision making. If AI had a home, it would be a building. The building would be the computer that makes decisions and operates the AI system.
    A building has a number of rooms that all have a job. Each room has a specific function, and the building is a single entity with many parts that all work together to perform their jobs.
    Building AI is like a building. It has a number of AI rooms and a computer system that operates the AI. The system is a single entity with many parts that all work together to perform their jobs.
    In the future


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What do you enjoy
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling city with a diverse population and a rich cultural heritage. It is the seat of the French government and is home to many famous landmarks and attractions. The city is also known for its fashion industry, with many famous designers and boutiques. Paris is a popular tourist destination and a major economic hub in Europe. It is a city that has a unique blend of history, culture, and modern
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends in AI that we can expect to see in the coming years:
    
    1. Increased automation: As AI continues to advance, we can expect to see more and more automation in various industries. This could include the automation of tasks such as data entry, customer service, and administrative work. As AI becomes more sophisticated, we can expect to see even more automation in areas such as manufacturing and transportation.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives
    


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
    Generated text:  [insert character's name], and I'm a/an [insert character's profession or role]. I specialize in [insert a specific skill or area of expertise] and have been working in this field for [insert number of years] years. I'm currently [insert a recent achievement, milestone, or accomplishment in your profession] and [insert any additional relevant information you feel should be included in your self-introduction]. I'm always up for learning and willing to adapt to new challenges, and I enjoy sharing my knowledge and experience with others. I'm excited to meet new people and see what new opportunities will be available to me in the future
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest and most populous city in the country. It was founded in 789 AD by Charlemagne and has been the seat of government and administration since the medieval period. Paris is known for its rich cultural history, including the Eiffel Tower, the Louvre Museum, and the Champs-Elysées. The city is also famous for its fashion industry, and is home to the world-renowned Paris Opera and Notre-Dame Cathedral. France's capital city, with its beautiful architecture and lively culture, is a major destination for both tourists and locals alike. It is often referred to as the "City of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to evolve rapidly, driven by advances in computing power, new data sources, and the growing complexity of human thought. Here are some potential trends that may shape the future of AI:
    
    1. Increased integration with human intelligence: As AI continues to become more advanced, it is expected to become more integrated with human intelligence. This could lead to more complex decision-making and decision-making processes that are more human-like.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there may be greater emphasis on ethical considerations. This could include considerations of bias, fairness, and transparency in AI algorithms.
    
    3. Greater


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

     am

     [

    Age

    ].

     I

     have

     always

     been

     an

     avid

     reader

    ,

     and

     I

     am

     always

     eager

     to

     learn

     new

     things

    .

     I

     also

     love

     to

     travel

    ,

     and

     I

     have

     explored

     many

     different

     countries

     around

     the

     world

    .

     My

     hobbies

     include

     writing

    ,

     playing

     board

     games

    ,

     and

     baking

    .

     What

     are

     some

     hobbies

     that

     you

     enjoy

    ?

     I

     also

     have

     a

     passion

     for

     food

    ,

     and

     I

     love

     to

     cook

     and

     enjoy

     the

     process

     of

     making

     new

     dishes

    .

     Thank

     you

     for

     asking

    ,

     and

     I

     hope

     to

     have

     the

     opportunity

     to

     share

     more

     about

     myself

     with

     you

    .

     Happy

     to

     meet

     you

    !

     How

     about

     you

    ?

     What

     are

     some

     hobbies

     you

     enjoy

    ?

     I

     enjoy

     playing

     board

     games

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     known

     as

     the

     "

    City

     of

     Love

    "

     for

     its

     romantic

     architecture

     and

     diverse

     cultural

     scene

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     population

     and

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

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     museums

    ,

     and

     art

     galleries

    ,

     including

     the

     Lou

    vre

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     Paris

     is

     also

     famous

     for

     its

     vibrant

     nightlife

    ,

     including

     the

     Mont

    mart

    re

     district

    ,

     and

     its

     annual

     E

    ly

    se

    es

     Festival

    ,

     which

     features

     music

    ,

     dance

    ,

     and

     performances

    .

     Despite

     its

     size

    ,

     Paris

     is

     a

     vibrant

     and

     dynamic

     city

     with

     a

     diverse

     range

     of

     people

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     significant

     advancements

     in

     areas

     such

     as

    :
    


    1

    .

     Enhanced

     natural

     language

     processing

    :

     AI

     systems

     will

     continue

     to

     improve

     in

     terms

     of

     accuracy

    ,

     efficiency

    ,

     and

     adapt

    ability

    ,

     allowing

     them

     to

     better

     understand

     and

     interpret

     natural

     language

    ,

     making

     it

     easier

     to

     create

     personalized

     experiences

     for

     users

    .
    


    2

    .

     Increased

     automation

    :

     AI

     systems

     will

     continue

     to

     automate

     tasks

     and

     processes

    ,

     increasing

     productivity

    ,

     efficiency

    ,

     and

     cost

     savings

    ,

     while

     also

     creating

     new

     opportunities

     for

     employment

    .
    


    3

    .

     AI

    -based

     personal

    ization

    :

     AI

     will

     continue

     to

     deliver

     personalized

     experiences

    ,

     using

     data

     to

     understand

     individual

     preferences

    ,

     habits

    ,

     and

     behavior

    ,

     and

     applying

     this

     information

     to

     create

     customized

     products

    ,

     services

    ,

     and

     interactions

    



```python
llm.shutdown()
```
