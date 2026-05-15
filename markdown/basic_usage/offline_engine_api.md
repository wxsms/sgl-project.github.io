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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.09it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.09it/s]


    2026-05-15 01:32:36,008 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 01:32:36] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.45it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.45it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.45it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.45it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.45it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.45it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.45it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.45it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.45it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.45it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.45it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.74it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.21it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.21it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.21it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.21it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  31%|███       | 18/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  31%|███       | 18/58 [00:00<00:01, 35.48it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 35.48it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:01, 35.48it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.51it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.51it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.51it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.51it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.51it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.51it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.74it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.68it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.23it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 44.43it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 39.39it/s]


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
    Generated text:  Eric. I am a 15-year-old boy. I like watching TV and playing computer games. I think watching TV is fun and interesting. I often go to the park with my friends and play games in it. However, one day last week, a terrible fire broke out in the park and made many people homeless and seriously hurt many people. I was so sad that I couldn't sleep well at night. I was so worried about the people who lost their homes that I couldn't go to school for a week. I felt like my whole world was turned upside down. I couldn't play games with my friends because my house
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is in charge of the country. He usually has a lot of work to do. He has to make important decisions, keep the country safe, and support the work of the people. He also has to make sure that people can go to work, have enough food and medicine, and take care of their children. Some of his work is very important. His job is very demanding. He works 12 hours a day, and he takes a lot of breaks. Most of the people don't like his job. They don't think he is fair and makes too many bad decisions. He is used to
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    
    B) London
    
    C) Madrid
    
    D) Rome
    
    E) Brussels
    
    To determine the capital of France, let's review the options provided:
    
    A) Paris
    B) London
    C) Madrid
    D) Rome
    E) Brussels
    
    The capital of France is Paris. Therefore, the correct answer is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  very bright, but the future of education is not. What will it look like? We are on the verge of a paradigm shift in the way we deliver education. With the rise of artificial intelligence (AI), it will be possible to automatically generate and analyse content, and even to deliver educational content. But this will also mean that we will have to use a variety of different ways to communicate with our students, and we may need to develop new ways of teaching.
    AI will be used in education to automatically generate and analyse content. AI will help to develop and improve the quality of content to be delivered in an automated way, and will also


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third-largest city in the world by population. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its fashion industry, with many famous designers and fashion houses operating in the area. Paris is a popular tourist destination, with millions of visitors each year. It is a cultural hub and a major economic center in Europe. The city is also known for its role in the French Revolution and its influence on French literature
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into various industries, from manufacturing to healthcare to transportation. This automation will likely lead to increased efficiency and productivity, but it will also lead to the loss of jobs for some workers.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be a need for ethical guidelines and regulations to ensure that AI is used in a responsible and fair manner. This will likely lead to increased
    


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
    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I've always had a talent for [Describe your specialty or interest]. I'm always eager to learn and grow, and am always up for adventure. I enjoy [What activity or hobby you have], and I'd love to share my passion with anyone interested. What's your name, and what's your occupation? [Name] I'm [Age] years old, and I've always been interested in [Occupation]. My specialty is [Describe your specialty or interest], and I'm always eager to learn and grow. I'm always up for adventure,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known as the "City of Light" for its towering architecture, vibrant culture, and historical significance. The city is situated in the south of France and has been a major center of politics, culture, and industry for over a millennium. Its unique blend of old-world charm and modern design has made Paris a popular destination for tourists and locals alike. It has hosted many important events, including the World Cup and the Olympic Games, and is considered one of the world's most beautiful cities. The city is also home to numerous art galleries, museums, and theaters, further enriching its cultural offerings. As of 20
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly advanced and diverse, with potential applications in many different areas. Here are some possible trends that could emerge in the field:
    
    1. Increased focus on ethical and social implications: As AI becomes more integrated into our lives, there will likely be an increased focus on ethical and social implications. This could lead to more transparency and accountability in AI development and use, and could also push for more responsible and ethical use of AI technology.
    
    2. Continued development of smaller, more capable AI systems: AI systems that can process information in smaller, more manageable chunks will likely become more common, allowing for faster and more accurate decision-making. This


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

    Character

    's

     Name

    ]

     and

     I

    'm

     here

     to

     share

     my

     skills

     with

     you

    .

     I

    'm

     a

     versatile

     professional

     who

     can

     help

     people

     solve

     complex

     problems

     and

     understand

     the

     unique

     challenges

     they

    're

     facing

    .

     Whether

     you

    're

     looking

     for

     a

     tool

     to

     streamline

     your

     operations

    ,

     a

     solution

     to

     a

     problem

     that

    's

     been

     gn

    aw

    ing

     at

     you

    ,

     or

     just

     someone

     to

     listen

     and

     share

     your

     thoughts

    ,

     I

    'm

     here

     to

     listen

     and

     help

     you

     find

     the

     right

     solutions

    .

     If

     you

     have

     any

     questions

    ,

     feel

     free

     to

     ask

    ,

     and

     I

    'll

     be

     here

     to

     answer

     them

    .

     I

    'm

     [

    Your

     Name

    ],

     and

     I

     look

     forward

     to

     working

     with

     you

    .

     #

    Self

    Introduction

     #

    Professional

     #

    Skill

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     located

     in

     the

     south

     of

     the

     country

     on

     the

     River

     Se

    ine

    .

     The

     city

     is

     home

     to

     many

     famous

     historical

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

     and

     the

     Lou

    vre

     Museum

    .

     It

     is

     also

     known

     for

     its

     vibrant

     and

     eclectic

     culture

    ,

     as

     well

     as

     its

     rich

     culinary

     traditions

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     has

     played

     an

     important

     role

     in

     French

     history

     and

     culture

     for

     centuries

    .

     The

     city

     is

     now

     home

     to

     over

     

    1

    7

     million

     people

     and

     is

     considered

     one

     of

     the

     largest

     cities

     in

     the

     world

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     hardware

    ,

     improvements

     in

     data

     analysis

    ,

     new

     applications

    ,

     and

     broader

     societal

     implications

    .

     Some

     potential

     trends

     include

    :
    


    1

    .

     Improved

     accuracy

     and

     reliability

    :

     As

     AI

     systems

     become

     more

     complex

     and

     sophisticated

    ,

     there

    's

     a

     greater

     chance

     they

    'll

     achieve

     even

     higher

     levels

     of

     accuracy

     and

     reliability

    .
    


    2

    .

     Enhanced

     privacy

     and

     security

    :

     AI

     systems

     will

     become

     even

     more

     sensitive

     and

     personal

    ,

     and

     there

     will

     be

     greater

     concerns

     about

     data

     privacy

     and

     security

    .
    


    3

    .

     AI

     integration

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     be

     integrated

     into

     a

     wide

     range

     of

     other

     technologies

    ,

     from

     healthcare

     and

     finance

     to

     transportation

     and

     manufacturing

    .
    


    4

    .

     More

    



```python
llm.shutdown()
```
