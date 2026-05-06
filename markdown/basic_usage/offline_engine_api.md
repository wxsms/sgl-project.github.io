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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.82it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.81it/s]


    2026-05-06 02:17:01,738 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 02:17:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.90it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.17it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.17it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.17it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.17it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.17it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.17it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.17it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.17it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.17it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.17it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:01, 19.79it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:01, 19.79it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:01, 19.79it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:01, 19.79it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:01, 19.79it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:01, 19.79it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:01, 19.79it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:01, 19.79it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:01, 19.79it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:01, 19.79it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 27.54it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 27.54it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 27.54it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 27.54it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 27.54it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 27.54it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 27.54it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 27.54it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 27.54it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 27.54it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 27.54it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 37.25it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 37.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.08it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=69.65 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=69.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.59 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.73it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.59 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.59 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=69.59 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=69.57 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s]

    Capturing num tokens (num_tokens=960 avail_mem=69.58 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s] Capturing num tokens (num_tokens=896 avail_mem=69.58 GB):  31%|███       | 18/58 [00:00<00:01, 34.74it/s]Capturing num tokens (num_tokens=896 avail_mem=69.58 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=832 avail_mem=69.58 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=768 avail_mem=69.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=704 avail_mem=69.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=640 avail_mem=69.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=576 avail_mem=69.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=576 avail_mem=69.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.40it/s]Capturing num tokens (num_tokens=512 avail_mem=69.55 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.40it/s]Capturing num tokens (num_tokens=480 avail_mem=69.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.40it/s]Capturing num tokens (num_tokens=448 avail_mem=69.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.40it/s]

    Capturing num tokens (num_tokens=416 avail_mem=69.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.40it/s]Capturing num tokens (num_tokens=384 avail_mem=69.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.40it/s]Capturing num tokens (num_tokens=384 avail_mem=69.56 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.32it/s]Capturing num tokens (num_tokens=352 avail_mem=69.56 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.32it/s]Capturing num tokens (num_tokens=320 avail_mem=69.55 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.32it/s]Capturing num tokens (num_tokens=288 avail_mem=69.55 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.32it/s]Capturing num tokens (num_tokens=256 avail_mem=69.55 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.32it/s]Capturing num tokens (num_tokens=240 avail_mem=69.54 GB):  57%|█████▋    | 33/58 [00:01<00:00, 42.32it/s]Capturing num tokens (num_tokens=240 avail_mem=69.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=224 avail_mem=69.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=208 avail_mem=69.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.67it/s]

    Capturing num tokens (num_tokens=192 avail_mem=69.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=176 avail_mem=69.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=160 avail_mem=69.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.67it/s]Capturing num tokens (num_tokens=160 avail_mem=69.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=144 avail_mem=69.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=128 avail_mem=69.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=112 avail_mem=69.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=96 avail_mem=69.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.31it/s] Capturing num tokens (num_tokens=80 avail_mem=69.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=80 avail_mem=69.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=64 avail_mem=69.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.16it/s]

    Capturing num tokens (num_tokens=48 avail_mem=69.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=32 avail_mem=69.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=28 avail_mem=69.46 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=24 avail_mem=69.46 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=24 avail_mem=69.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=20 avail_mem=69.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=16 avail_mem=69.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=12 avail_mem=69.45 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=8 avail_mem=69.45 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.77it/s] Capturing num tokens (num_tokens=4 avail_mem=69.44 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.77it/s]Capturing num tokens (num_tokens=4 avail_mem=69.44 GB): 100%|██████████| 58/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=4 avail_mem=69.44 GB): 100%|██████████| 58/58 [00:01<00:00, 38.94it/s]


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
    Generated text:  Tom and I am currently studying for my accounting exam. I'm having a hard time understanding how to determine the correct expense category for a particular transaction. Can you please provide me with some examples and explain them to me? Sure, I'd be happy to help! Here are a few examples of expenses and how to determine which category they should be categorized under:
    
    1. Food expense - The cost of buying food for a restaurant, such as burgers or fries, would be classified as a food expense. This is because the restaurant is spending money on the ingredients and labor to make the food and the customers are buying it as a product.
    
    2
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to become more environmentally conscious. He decided to increase the speed of the traffic police, and the traffic police has the responsibility to control traffic in the city. After a certain number of years, the speed of the traffic police increased by 20%. If the speed of the traffic police increased by 20%, what is the percentage increase of the speed of the traffic police? 
    Answer Choices: (A) 10% (B) 20% (C) 30% (D) 40% (E) 50%
    1. Let the initial speed of the traffic police be den
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. London C. New York D. Tokyo
    A
    
    What is the purpose of the initiative? A. To create an exciting scene B. To create a sense of urgency C. To create a sense of fascination D. To create a sense of puzzlement
    A
    
    Which of the following is an important study of the relationship between personal and professional life? A. Ethics of business B. Economic order C. Business ethics D. Corporate politics
    C
    
    What is the most important objective of the initiative? A. To create an exciting scene B. To create a sense of urgency C. To create a
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, and the industry is moving to prepare for the challenges it faces. The AI industry is continuously evolving, with the ability to complete tasks more quickly and accurately than humans. This leads to a need for new ways of thinking and a growth in the need for new skills.
    
    The future of AI will be one of growth, and we will see an increase in the number of jobs created by AI. AI is creating new industries and will create a lot of new jobs. The jobs that will be created by AI include but are not limited to, data scientists, computer engineers, and software developers.
    
    It is important for people to understand the impact


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm passionate about [reason for job title], and I'm always looking for ways to [action or goal]. I'm a [character trait or quality] and I'm always [positive or negative] about my work. I'm [character description] and I'm always [positive or negative] about my work. I'm [character trait or quality] and I'm always [positive or negative] about my work. I'm [character description] and I'm always [positive or negative] about my work. I'm [character trait or quality]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French language. It is also a popular tourist destination, attracting millions of visitors each year. The city is home to many famous French artists, writers, and musicians, and is known for its cuisine, including its famous French fries. Paris is a vibrant and dynamic city with a rich history and a vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into various industries, leading to the automation of tasks that were previously done by humans. This could result in job losses in some sectors, but also create new opportunities for people to work in areas like data analysis, machine learning, and robotics.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be a growing need for privacy and security measures to protect personal data. This
    


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
    Generated text:  [Name], and I'm a/an [Job Title] with [Number of Years in Industry] years experience in [Industry]. I'm a[Type of Person] with [Strengths and Weaknesses] who always strive to [Motivation] and enjoy [Interests and hobbies]. I enjoy [Job Hobby], [Sports, Music, Reading, etc.], and [Other Interests, if applicable]. If you're reading this, you're the right person to talk to about [Your Profession], [Your Position], or [Your Background]. [Name], you're the one I'm thinking of. Let's get to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France, the second largest city in the European Union and the 19th largest city on Earth. It is home to millions of people and is the cultural, economic and political center of the country. The city has a rich history dating back to ancient times, including ancient Greek and Roman ruins and the 19th-century Exposition Universelle. Paris is known for its architecture, cuisine, music, and fashion, and is a popular tourist destination. The city is also famous for its fashion industry, which includes brands such as Chanel and Louis Vuitton. The city is often referred to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be a highly dynamic and multifaceted field, with a wide range of possibilities and possibilities. Some of the potential trends in AI include:
    
    1. Increased integration of AI into human workplaces: As AI becomes more advanced and integrated into everyday work, it is likely that it will become even more prevalent in the workplace. This could lead to more automation, improved productivity, and increased efficiency.
    
    2. Development of AI-powered systems for environmental and social issues: As AI is used to monitor and mitigate the impact of environmental and social issues, such as climate change and social inequality, it is likely that this technology will continue to evolve and improve


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

    ],

     and

     I

    'm

     [

    Your

     Occupation

    /

    Role

    ].

     I

     have

     a

     passion

     for

     [

    Your

     Hobby

    /

    Interest

    ]

     and

     I

    'm

     always

     looking

     to

     learn

     new

     things

     and

     grow

     as

     a

     person

    .

     I

    'm

     very

     open

    -minded

     and

     enjoy

     sharing

     my

     thoughts

     and

     experiences

     with

     others

    .

     What

    's

     your

     favorite

     hobby

    /

    interest

     and

     how

     have

     you

     been

     enjoying

     it

     lately

    ?


    Hi

     there

    !

     My

     name

     is

     [

    Your

     Name

    ],

     and

     I

    'm

     a

     [

    Your

     Occupation

    /

    Role

    ].

     I

     love

     [

    Your

     Hobby

    /

    Interest

    ],

     and

     I

    'm

     always

     up

     for

     new

     adventures

    !

     I

    'm

     very

     open

    -minded

     and

     enjoy

     sharing

     my

     thoughts

     and

     experiences

     with

     others

    .

     What

    's

     your

     favorite

     hobby

    /

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Incorrect

    .

     The

     capital

     of

     France

     is

     not

     Paris

    .

     The

     capital

     of

     France

     is

     Rome

    .

     The

     correct

     answer

     is

    :

     Paris

     is

     the

     capital

     of

     France

    .

     The

     correct

     answer

     is

    :

     Paris

     is

     the

     capital

     of

     France

    .

     The

     correct

     answer

     is

    :

     Paris

     is

     the

     capital

     of

     France

    .

     The

     correct

     answer

     is

    :

     Paris

     is

     the

     capital

     of

     France

    .

     The

     correct

     answer

     is

    :

     Paris

     is

     the

     capital

     of

     France

    .

     The

     correct

     answer

     is

    :

     Paris

     is

     the

     capital

     of

     France

    .

     The

     correct

     answer

     is

    :

     Paris

     is

     the

     capital

     of

     France

    .

     The

     correct

     answer

     is

    :

     Paris

     is

     the

     capital

     of

     France

    .

     The

     correct

     answer

     is

    :

     Paris

     is

     the

     capital

     of

     France

    .

     The

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     see

     continued

     growth

     and

     development

    ,

     driven

     by

     advances

     in

     computer

     science

    ,

     machine

     learning

    ,

     and

     the

     increasing

     availability

     of

     data

    .

     Some

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     everyday

     life

    :

     This

     could

     involve

     more

     widespread

     adoption

     of

     AI

    -powered

     technologies

     like

     voice

     assistants

    ,

     self

    -driving

     cars

    ,

     and

     smart

     home

     devices

    .

     AI

     will

     also

     play

     a

     more

     significant

     role

     in

     healthcare

    ,

     education

    ,

     and

     transportation

    .
    


    2

    .

     Improved

     privacy

     and

     data

     security

    :

     As

     AI

     becomes

     more

     prevalent

     in

     everyday

     life

    ,

     there

     will

     be

     increasing

     pressure

     to

     ensure

     that

     personal

     data

     is

     protected

    .

     This

     could

     lead

     to

     more

     robust

     privacy

     regulations

    ,

     greater

     use

     of

     encryption

    ,

     and

     enhanced

    



```python
llm.shutdown()
```
