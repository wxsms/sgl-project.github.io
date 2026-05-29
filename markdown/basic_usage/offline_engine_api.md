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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:07,  4.33s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:07,  4.33s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.42it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.42it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.42it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.42it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.42it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.42it/s]

    Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.42it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 20.56it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 20.56it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 20.56it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 20.56it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 20.56it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 20.56it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:01, 20.56it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:01, 20.56it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:01, 20.56it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:01, 20.56it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 28.82it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 28.82it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 28.82it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 38.89it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 38.89it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 38.89it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 38.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:04, 13.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:04, 13.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:04, 13.87it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:04, 13.87it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:03, 16.24it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  21%|██        | 12/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:02, 21.66it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.09it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  31%|███       | 18/58 [00:01<00:02, 16.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  31%|███       | 18/58 [00:01<00:02, 16.53it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  31%|███       | 18/58 [00:01<00:02, 16.53it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.07it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.07it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.07it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.07it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.54it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.54it/s]

    Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.54it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.54it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:01<00:01, 19.54it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.95it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.95it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.95it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.95it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.95it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.95it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.65it/s]

    Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.07it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.07it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.07it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.07it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.07it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.07it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.57it/s]

    Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.57it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.63it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.30it/s]

    Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:02<00:00, 41.30it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:02<00:00, 41.30it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  91%|█████████▏| 53/58 [00:02<00:00, 41.30it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:02<00:00, 42.58it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:02<00:00, 27.99it/s]


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
    Generated text:  Giovanni. And I'm here on my holiday in South Africa. And I'm hoping to get to know more about the culture and the people here. And I really love the nightlife and the excitement of the country. And there's so much to see and to do. I'm really looking forward to some good food, some good drinks and to some activities. And I'm hoping to spend some time with friends. My friend Alain is from France and he has a lot of friends in South Africa. He's really good at going out and meeting people. He is interested in the history of the country and the culture of South Africa.
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. This is what he has to say.
    
    ```
    Hello, Mr. President.
    
    This is a time of great political change in our country. To understand it all, I have to show you some of the facts.
    First of all, let’s look at the budget. The total bill for our government in this year is $400 billion. The federal budget for the fiscal year 2019 is $418 billion.
    The amount is getting a little higher in the first few years, but that’s because the previous year’s numbers were a bit low. The 2018
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. Marseille
    C. Lyon
    D. London
    Answer:
    
    A
    
    On a certain island, there are 200 students. Half of these students are boys, and the rest are girls. If 10 more boys join the group, the number of boys will increase by 10. How many girls are there on the island now? 
    A. 300
    B. 200
    C. 400
    D. 350
    Answer:
    
    D
    
    The rapid development of the automobile industry has led to a severe shortage of spare
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it isn’t inherently safe. In fact, many of the world’s most advanced AI systems can be hacked and abused.
    AI systems can be used for bad things. To prevent that, engineers must be mindful of the AI they create and must take steps to ensure that AI is safe.
    One thing engineers can do is to make sure that the data that they use to create their AI systems is accurate and up-to-date. They should also ensure that the data they use to train their AI systems is unbiased and doesn’t contain any potential biases.
    Another thing engineers can do is to make sure that the AI systems they create have clear


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


    Generated text:  [Name] and I'm a [occupation] who has been [number of years] in the industry. I'm passionate about [reason for passion], and I'm always looking for ways to [action or achievement]. I'm a [type of person] who is [positive or negative] about life, and I'm always ready to [action or achievement]. I'm [type of person] who is [positive or negative] about life, and I'm always ready to [action or achievement]. I'm [type of person] who is [positive or negative] about life, and I'm always ready to [action or achievement
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and culture. It is located on the Seine River and is the largest city in France by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also famous for its fashion industry, art, and cuisine. Paris is a popular tourist destination and a cultural hub for France. It is home to many famous museums, theaters, and restaurants. The city is also known for its annual festivals and events, such as the Eiffel Tower Parade and the Paris Fashion
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends in AI that are expected to shape the future:
    
    1. Increased automation: One of the most significant trends in AI is the increasing automation of tasks that are currently done by humans. This could include tasks such as data analysis, decision-making, and routine maintenance. As AI becomes more advanced, it is likely to be able to perform these tasks more efficiently and accurately than humans.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there is
    


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
    Generated text:  [insert character name], and I'm [insert character's age and gender]. I'm a [insert occupation or profession]. I have [insert favorite hobby or interest]. I'm passionate about [insert reason why I love [occupation/profession]>]. And [insert the most recent project or achievement you've achieved]. What's your story? 
    
    (Words can be cut to a maximum of 15 characters)
    
    Certainly! Here's a neutral self-introduction for your fictional character:
    
    **"Hello, my name is [insert character name], and I'm [insert character's age and gender]. I'm a [insert occupation or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city and the most populous metropolitan area in France and the European Union. It is located on the left bank of the Seine River in the Île-de-France region, near the Atlantic Ocean. Paris is considered one of the world's most important cultural, artistic, literary, and scientific centers, and is known worldwide for its monuments, architecture, cuisine, fashion, and fine arts. It is the most visited city in the world and is home to numerous museums, theaters, opera houses, and other cultural institutions. Paris is also a major financial center and the birthplace of many famous artists, writers,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly dynamic and multifaceted, with a wide range of potential trends that could shape the direction of progress in this field. Here are some of the most likely trends that could lead to further innovations in AI:
    
    1. Improved ethics and safety measures: As AI systems become more integrated into our daily lives, they are likely to require greater attention to their ethical and safety concerns. This could lead to new ethical guidelines and regulations to govern the development and deployment of AI technologies.
    
    2. Increased availability of AI-driven products and services: As AI technology continues to improve and become more ubiquitous, we may see more widespread adoption of AI-driven


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

    insert

     fictional

     character

    's

     name

    ].

     I

     am

     a

     [

    insert

     fictional

     character

    's

     age

    ]

     year

     old

    ,

     [

    insert

     fictional

     character

    's

     profession

     or

     hobby

    ].

     I

     come

     from

     [

    insert

     fictional

     character

    's

     home

     country

     or

     background

    ].

     I

     am

     [

    insert

     fictional

     character

    's

     personality

     or

     unique

     trait

    ].

     I

     enjoy

     [

    insert

     fictional

     character

    's

     interests

     or

     hobbies

    ].

     I

     am

     a

     [

    insert

     fictional

     character

    's

     favorite

     book

     or

     TV

     show

    ].

     I

     am

     a

     [

    insert

     fictional

     character

    's

     occupation

     or

     profession

    ].

     I

     strive

     to

     be

     a

     [

    insert

     fictional

     character

    's

     character

     trait

     or

     leadership

     style

    ].

     I

     am

     always

     [

    insert

     fictional

     character

    's

     favorite

     ad

    jectives

    ].

     I

     am

     [

    insert

     fictional

     character

    's

     favorite

     quote

    
    
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

     France

     by

     population

     and

     is

     the

     seat

     of

     government

     and

     most

     important

     institution

     of

     the

     country

    .

     The

     city

     is

     home

     to

     many

     historical

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     It

     is

     also

     known

     for

     its

     rich

     cultural

     heritage

    ,

     including

     its

     famous

     museums

    ,

     theaters

    ,

     and

     art

     galleries

    .

     Paris

     is

     a

     bustling

     city

     with

     a

     diverse

     population

     and

     is

     a

     major

     economic

     and

     political

     center

     of

     Europe

    .

     It

     is

     the

     unofficial

     capital

     of

     the

     French

     Republic

    ,

     which

     is

     the

     capital

     of

     France

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     home

     to

     many

     iconic

     landmarks

     and

     museums

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     an

     exciting

     and

     rapidly

     evolving

     area

    .

     Some

     of

     the

     trends

     we

     can

     expect

     to

     see

     in

     AI

     include

    :
    


    1

    .

     Increased

     automation

    :

     With

     the

     growth

     of

     automation

     in

     industries

     like

     manufacturing

    ,

     transportation

    ,

     and

     service

    ,

     we

     can

     expect

     AI

     to

     become

     more

     prevalent

     in

     jobs

    .

     This

     could

     lead

     to

     the

     creation

     of

     a

     new

     type

     of

     "

    human

    "

     -

     an

     "

    AI

     human

    "

     who

     can

     perform

     tasks

     with

     the

     same

     level

     of

     accuracy

     and

     efficiency

     as

     a

     human

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

     in

     healthcare

     to

     improve

     patient

     outcomes

     and

     prevent

     diseases

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

     more

     widespread

     use

     of

     AI

     in

     healthcare

    .
    


    



```python
llm.shutdown()
```
